//
//  Gemma4VLM.swift
//  mlx-swift-lm
//
//  Gemma 4 unified VLM (text + vision). Audio support deferred.
//
//  Ported from:
//    - Blaizzy/mlx-vlm gemma4/{gemma4.py, vision.py, processing_gemma4.py, config.py, language.py}
//    - ml-explore/mlx-lm mlx_lm/models/gemma4_text.py (for text decoder — already exists as Gemma4Text.swift in MLXLLM)
//
//  Architecture summary:
//    - language_model: Gemma4TextModel (from MLXLLM) — PLE, dual KV cache, shared KV layers, dual RoPE
//    - vision_tower: custom Gemma4 vision encoder (SigLIP-like w/ 2D RoPE + learned position table + ClippableLinear)
//    - embed_vision: MultimodalEmbedder (RMSNormNoScale + Linear projection)
//    - Phase 3 DROPS audio_tower/embed_audio weights in sanitize() for memory savings.
//
//  The file is strictly self-contained except for:
//    1. import MLXLLM (to reuse Gemma4TextConfiguration + Gemma4TextModel)
//    2. a 2-line registry entry in VLMModelFactory.swift
//    3. a small additive public method `callWithEmbeddings` on Gemma4TextModel
//
//  This design keeps the path to upstream ml-explore/mlx-swift-lm clean: when upstream
//  ships Gemma 4 VLM we can drop this file + revert those 3 touches with zero merge conflict.
//

import CoreImage
import Foundation
import MLX
import MLXFast
import MLXLLM
import MLXLMCommon
import MLXNN
import Tokenizers

// MARK: - Errors

private enum Gemma4VLMError: LocalizedError {
    case imageSizeInvalid(String)
    case tokenExpansionFailed(String)

    var errorDescription: String? {
        switch self {
        case .imageSizeInvalid(let msg): return "Gemma4VLM image size invalid: \(msg)"
        case .tokenExpansionFailed(let msg): return "Gemma4VLM token expansion failed: \(msg)"
        }
    }
}

// MARK: - Helper modules

/// RMSNorm without a learnable scale parameter (Gemma 4 multimodal projector uses this).
private class RMSNormNoScale: Module, UnaryLayer {
    let eps: Float
    init(eps: Float = 1e-6) {
        self.eps = eps
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let variance = (x * x).mean(axis: -1, keepDims: true)
        return x * MLX.rsqrt(variance + eps)
    }
}

/// Linear layer with optional input/output activation clamping.
/// Mirrors PyTorch's `Gemma4ClippableLinear`: wraps an `MLXNN.Linear`, applies `mx.clip`
/// at input and output. Clip buffers are loaded from the checkpoint as 4 scalar MLXArrays
/// (input_min, input_max, output_min, output_max). When `useClipping` is false the clip
/// buffers are NOT created — the module degrades to a plain Linear.
private class Gemma4ClippableLinear: Module, UnaryLayer {
    @ModuleInfo var linear: Linear

    @ParameterInfo(key: "input_min") var inputMin: MLXArray
    @ParameterInfo(key: "input_max") var inputMax: MLXArray
    @ParameterInfo(key: "output_min") var outputMin: MLXArray
    @ParameterInfo(key: "output_max") var outputMax: MLXArray

    let useClipping: Bool

    init(_ inputDim: Int, _ outputDim: Int, bias: Bool = false, useClipping: Bool = true) {
        self._linear.wrappedValue = Linear(inputDim, outputDim, bias: bias)
        self.useClipping = useClipping
        // Initialize clip bounds to ±infinity so that the clamp is a no-op until the
        // real checkpoint buffers are loaded. When `useClipping=false` these arrays
        // are still constructed but never consulted on the hot path.
        self._inputMin.wrappedValue = MLXArray(-Float.infinity)
        self._inputMax.wrappedValue = MLXArray(Float.infinity)
        self._outputMin.wrappedValue = MLXArray(-Float.infinity)
        self._outputMax.wrappedValue = MLXArray(Float.infinity)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = x
        if useClipping {
            y = clip(y, min: inputMin, max: inputMax)
        }
        y = linear(y)
        if useClipping {
            y = clip(y, min: outputMin, max: outputMax)
        }
        return y
    }
}

// MARK: - 2D RoPE for vision encoder

/// Rotates the second half of the head dimension by negating it and swapping with the
/// first half, matching PyTorch's `rotate_half`.
private func rotateHalf(_ x: MLXArray) -> MLXArray {
    let lastDim = x.shape.last!
    let half = lastDim / 2
    let x1 = x[.ellipsis, 0 ..< half]
    let x2 = x[.ellipsis, half ..< lastDim]
    return concatenated([-x2, x1], axis: -1)
}

/// Apply multidimensional RoPE. `inputs` is `[B, N, L, H]` (queries or keys after QK-norm
/// and heads reshape). `positions` is `[B, L, ndim]` giving per-token integer positions
/// along `ndim` spatial axes. The head dimension `H` is split into `ndim` equal slabs;
/// slab `d` is rotated by `positions[..., d]` using the given `baseFrequency`.
///
/// Critical: `rotate_half` must NOT mix channels across spatial dimensions, so we process
/// each slab independently and concatenate the results.
private func applyMultidimensionalRoPE(
    _ inputs: MLXArray,
    positions: MLXArray,
    baseFrequency: Float = 100.0
) -> MLXArray {
    let headDim = inputs.shape.last!
    let ndim = positions.shape.last!
    let channelsPerDim = 2 * (headDim / (2 * ndim))
    let halfPerDim = channelsPerDim / 2

    var parts: [MLXArray] = []
    parts.reserveCapacity(ndim)

    for d in 0 ..< ndim {
        let xPart = inputs[.ellipsis, (d * channelsPerDim) ..< ((d + 1) * channelsPerDim)]

        // freq_exponents = (2.0 / channels_per_dim) * arange(0, half_per_dim)
        var freqExponents = MLXArray(0 ..< halfPerDim).asType(.float32)
        freqExponents = freqExponents * Float(2.0 / Double(channelsPerDim))
        let timescale = MLX.pow(MLXArray(baseFrequency), freqExponents)

        // positions[..., d:d+1]  ->  [B, L, 1]
        let posD = positions[.ellipsis, d ..< (d + 1)].asType(.float32)

        // sinusoid = pos / timescale  ->  [B, L, half_per_dim]
        let sinusoid = posD / timescale

        var cosD = MLX.cos(sinusoid)
        var sinD = MLX.sin(sinusoid)
        // Concatenate twice to cover both halves of the rotate_half operation.
        cosD = concatenated([cosD, cosD], axis: -1).asType(inputs.dtype)
        sinD = concatenated([sinD, sinD], axis: -1).asType(inputs.dtype)
        // Insert the head axis so broadcast aligns: [B, 1, L, C]
        cosD = expandedDimensions(cosD, axis: 1)
        sinD = expandedDimensions(sinD, axis: 1)

        let yPart = xPart * cosD + rotateHalf(xPart) * sinD
        parts.append(yPart)
    }

    return concatenated(parts, axis: -1)
}

// MARK: - Configurations

public struct Gemma4VisionConfiguration: Codable, Sendable {
    public let modelType: String?
    public let hiddenSize: Int?
    public let intermediateSize: Int?
    public let numHiddenLayers: Int?
    public let numAttentionHeads: Int?
    public let numKeyValueHeads: Int?
    public let headDim: Int?
    public let patchSize: Int?
    public let positionEmbeddingSize: Int?
    public let poolingKernelSize: Int?
    public let defaultOutputLength: Int?
    public let useClippedLinears: Bool?
    public let standardize: Bool?
    public let rmsNormEps: Float?
    public let ropeParameters: RopeParameters?

    public struct RopeParameters: Codable, Sendable {
        public let ropeTheta: Float?
        enum CodingKeys: String, CodingKey {
            case ropeTheta = "rope_theta"
        }
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case patchSize = "patch_size"
        case positionEmbeddingSize = "position_embedding_size"
        case poolingKernelSize = "pooling_kernel_size"
        case defaultOutputLength = "default_output_length"
        case useClippedLinears = "use_clipped_linears"
        case standardize
        case rmsNormEps = "rms_norm_eps"
        case ropeParameters = "rope_parameters"
    }

    // Computed defaults so downstream code never sees a missing value.
    public var effectiveHiddenSize: Int { hiddenSize ?? 768 }
    public var effectiveIntermediateSize: Int { intermediateSize ?? 3072 }
    public var effectiveNumHiddenLayers: Int { numHiddenLayers ?? 16 }
    public var effectiveNumAttentionHeads: Int { numAttentionHeads ?? 12 }
    public var effectiveNumKeyValueHeads: Int { numKeyValueHeads ?? 12 }
    public var effectiveHeadDim: Int { headDim ?? 64 }
    public var effectivePatchSize: Int { patchSize ?? 16 }
    public var effectivePositionEmbeddingSize: Int { positionEmbeddingSize ?? 10240 }
    public var effectivePoolingKernelSize: Int { poolingKernelSize ?? 3 }
    public var effectiveDefaultOutputLength: Int { defaultOutputLength ?? 280 }
    public var effectiveUseClippedLinears: Bool { useClippedLinears ?? false }
    public var effectiveStandardize: Bool { standardize ?? false }
    public var effectiveRmsNormEps: Float { rmsNormEps ?? 1e-6 }
    public var effectiveRopeTheta: Float { ropeParameters?.ropeTheta ?? 100.0 }
}

public struct Gemma4VLMConfiguration: Codable, Sendable {
    public let textConfiguration: Gemma4TextConfiguration
    public let visionConfiguration: Gemma4VisionConfiguration
    public let modelType: String?

    // Multimodal token IDs — all optional, with computed effective* getters for defaults
    public let imageTokenId: Int?
    public let audioTokenId: Int?
    public let videoTokenId: Int?
    public let boiTokenId: Int?
    public let eoiTokenId: Int?
    public let boaTokenId: Int?
    public let eoaTokenId: Int?
    public let padTokenId: Int?
    public let visionSoftTokensPerImage: Int?
    public let quantization: BaseConfiguration.Quantization?

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case modelType = "model_type"
        case imageTokenId = "image_token_id"
        case audioTokenId = "audio_token_id"
        case videoTokenId = "video_token_id"
        case boiTokenId = "boi_token_id"
        case eoiTokenId = "eoi_token_id"
        case boaTokenId = "boa_token_id"
        case eoaTokenId = "eoa_token_id"
        case padTokenId = "pad_token_id"
        case visionSoftTokensPerImage = "vision_soft_tokens_per_image"
        case quantization
    }

    public var effectiveImageTokenId: Int { imageTokenId ?? 258880 }
    public var effectiveBoiTokenId: Int { boiTokenId ?? 255999 }
    public var effectiveEoiTokenId: Int { eoiTokenId ?? 258882 }
    public var effectivePadTokenId: Int { padTokenId ?? 0 }
    public var effectiveVisionSoftTokensPerImage: Int { visionSoftTokensPerImage ?? 280 }
}

public struct Gemma4ProcessorConfiguration: Codable, Sendable {
    public let processorClass: String?
    public let imageProcessor: ImageProcessor?

    public struct ImageSize: Codable, Sendable {
        public let height: Int
        public let width: Int
    }

    public struct ImageProcessor: Codable, Sendable {
        public let imageProcessorType: String?
        public let imageMean: [CGFloat]?
        public let imageStd: [CGFloat]?
        public let doNormalize: Bool?
        public let doRescale: Bool?
        public let doResize: Bool?
        public let doConvertRgb: Bool?
        public let rescaleFactor: Float?
        public let patchSize: Int?
        public let poolingKernelSize: Int?
        public let imageSeqLength: Int?
        public let maxSoftTokens: Int?
        public let size: ImageSize?

        enum CodingKeys: String, CodingKey {
            case imageProcessorType = "image_processor_type"
            case imageMean = "image_mean"
            case imageStd = "image_std"
            case doNormalize = "do_normalize"
            case doRescale = "do_rescale"
            case doResize = "do_resize"
            case doConvertRgb = "do_convert_rgb"
            case rescaleFactor = "rescale_factor"
            case patchSize = "patch_size"
            case poolingKernelSize = "pooling_kernel_size"
            case imageSeqLength = "image_seq_length"
            case maxSoftTokens = "max_soft_tokens"
            case size
        }
    }

    enum CodingKeys: String, CodingKey {
        case processorClass = "processor_class"
        case imageProcessor = "image_processor"
    }

    // Computed convenience getters (all fall back to sensible Gemma 4 defaults).
    public var effectivePatchSize: Int { imageProcessor?.patchSize ?? 16 }
    public var effectivePoolingKernelSize: Int { imageProcessor?.poolingKernelSize ?? 3 }
    public var effectiveMaxSoftTokens: Int {
        imageProcessor?.maxSoftTokens ?? imageProcessor?.imageSeqLength ?? 280
    }
    public var effectiveRescaleFactor: Float { imageProcessor?.rescaleFactor ?? (1.0 / 255.0) }
    public var effectiveDoNormalize: Bool { imageProcessor?.doNormalize ?? false }
    public var effectiveImageMean: (CGFloat, CGFloat, CGFloat) {
        let m = imageProcessor?.imageMean ?? [0.0, 0.0, 0.0]
        return m.count >= 3 ? (m[0], m[1], m[2]) : (0.0, 0.0, 0.0)
    }
    public var effectiveImageStd: (CGFloat, CGFloat, CGFloat) {
        let s = imageProcessor?.imageStd ?? [1.0, 1.0, 1.0]
        return s.count >= 3 ? (s[0], s[1], s[2]) : (1.0, 1.0, 1.0)
    }
}

// MARK: - Vision encoder submodules

/// Converts pixel tensor [B, C, H, W] into patch tokens via reshape + per-channel normalization
/// into [-1, 1] + linear projection, then adds learned 2D position embeddings via one-hot lookup.
private class Gemma4VisionPatchEmbedder: Module {
    // Note: patch_embedder.input_proj in the checkpoint is a *plain* Linear with the
    // weight stored at `vision_tower.patch_embedder.input_proj.weight` (NOT
    // `...input_proj.linear.weight`). The encoder layers use ClippableLinear with
    // an extra `linear.` namespace level, but this first projection does not.
    @ModuleInfo(key: "input_proj") var inputProj: Linear
    @ParameterInfo(key: "position_embedding_table") var positionEmbeddingTable: MLXArray

    let patchSize: Int
    let hiddenSize: Int
    let positionEmbeddingSize: Int

    init(_ config: Gemma4VisionConfiguration) {
        let p = config.effectivePatchSize
        let h = config.effectiveHiddenSize
        let pes = config.effectivePositionEmbeddingSize
        self.patchSize = p
        self.hiddenSize = h
        self.positionEmbeddingSize = pes

        let patchFeatureDim = 3 * p * p
        self._inputProj.wrappedValue = Linear(patchFeatureDim, h, bias: false)

        // Position embedding table: [2, position_embedding_size, hidden_size]
        // One row per spatial axis; patch (row, col) gets embeddings from row 0 at col index
        // and row 1 at row index, summed.
        self._positionEmbeddingTable.wrappedValue = MLXArray.zeros([2, pes, h])
        super.init()
    }

    /// - Parameters:
    ///   - pixelValues: `[B, C, H, W]`, values already in [0, 1] (do_rescale applied by processor)
    ///   - patchPositions: `[B, L, 2]` int positions per patch (col, row)
    /// - Returns: `[B, L, hiddenSize]` patch embeddings with positions added.
    func callAsFunction(_ pixelValues: MLXArray, patchPositions: MLXArray) -> MLXArray {
        let B = pixelValues.dim(0)
        let C = pixelValues.dim(1)
        let H = pixelValues.dim(2)
        let W = pixelValues.dim(3)
        let p = patchSize
        let pH = H / p
        let pW = W / p

        // [B, C, pH, p, pW, p]  ->  [B, pH, pW, p, p, C]  ->  [B, pH*pW, C*p*p]
        var patches = pixelValues.reshaped(B, C, pH, p, pW, p)
        patches = patches.transposed(0, 2, 4, 3, 5, 1)
        patches = patches.reshaped(B, pH * pW, C * p * p)

        // Manual [0, 1] -> [-1, 1] re-centering (matches Python `2 * (x - 0.5)`).
        patches = 2.0 * (patches - 0.5)

        var embeds = inputProj(patches)

        // Learned 2D position embeddings via one-hot indexing.
        // position_embedding_table has shape [2, position_embedding_size, hidden_size].
        // For axis 0 (col) use patchPositions[..., 0]; for axis 1 (row) use patchPositions[..., 1].
        // Each is a gather; we sum both contributions onto the patch embeddings.
        let colIdx = patchPositions[.ellipsis, 0]
        let rowIdx = patchPositions[.ellipsis, 1]

        let tableCol = positionEmbeddingTable[0]  // [position_embedding_size, hidden_size]
        let tableRow = positionEmbeddingTable[1]

        let colEmbed = tableCol[colIdx]  // [B, L, hidden_size]
        let rowEmbed = tableRow[rowIdx]

        embeds = embeds + colEmbed + rowEmbed
        return embeds
    }
}

private class Gemma4VisionAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Gemma4ClippableLinear
    @ModuleInfo(key: "k_proj") var kProj: Gemma4ClippableLinear
    @ModuleInfo(key: "v_proj") var vProj: Gemma4ClippableLinear
    @ModuleInfo(key: "o_proj") var oProj: Gemma4ClippableLinear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float
    let ropeBase: Float

    init(_ config: Gemma4VisionConfiguration) {
        let nH = config.effectiveNumAttentionHeads
        let nKV = config.effectiveNumKeyValueHeads
        let hD = config.effectiveHeadDim
        let dim = config.effectiveHiddenSize
        let eps = config.effectiveRmsNormEps
        self.numHeads = nH
        self.numKVHeads = nKV
        self.headDim = hD
        self.scale = 1.0  // Python uses scale=1.0 (no 1/sqrt(head_dim) — matches Blaizzy)
        self.ropeBase = config.effectiveRopeTheta

        let useClip = config.effectiveUseClippedLinears

        self._qProj.wrappedValue = Gemma4ClippableLinear(dim, nH * hD, bias: false, useClipping: useClip)
        self._kProj.wrappedValue = Gemma4ClippableLinear(dim, nKV * hD, bias: false, useClipping: useClip)
        self._vProj.wrappedValue = Gemma4ClippableLinear(dim, nKV * hD, bias: false, useClipping: useClip)
        self._oProj.wrappedValue = Gemma4ClippableLinear(nH * hD, dim, bias: false, useClipping: useClip)

        self._qNorm.wrappedValue = RMSNorm(dimensions: hD, eps: eps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: hD, eps: eps)

        super.init()
    }

    func callAsFunction(_ x: MLXArray, patchPositions: MLXArray) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        var q = qProj(x).reshaped(B, L, numHeads, headDim)
        var k = kProj(x).reshaped(B, L, numKVHeads, headDim)
        var v = vProj(x).reshaped(B, L, numKVHeads, headDim)

        q = qNorm(q)
        k = kNorm(k)

        // [B, L, N, H] -> [B, N, L, H]
        q = q.transposed(0, 2, 1, 3)
        k = k.transposed(0, 2, 1, 3)
        v = v.transposed(0, 2, 1, 3)

        // Apply 2D RoPE to Q and K
        q = applyMultidimensionalRoPE(q, positions: patchPositions, baseFrequency: ropeBase)
        k = applyMultidimensionalRoPE(k, positions: patchPositions, baseFrequency: ropeBase)

        let output = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: .none
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return oProj(output)
    }
}

private class Gemma4VisionMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Gemma4ClippableLinear
    @ModuleInfo(key: "up_proj") var upProj: Gemma4ClippableLinear
    @ModuleInfo(key: "down_proj") var downProj: Gemma4ClippableLinear

    init(_ config: Gemma4VisionConfiguration) {
        let d = config.effectiveHiddenSize
        let h = config.effectiveIntermediateSize
        let useClip = config.effectiveUseClippedLinears
        self._gateProj.wrappedValue = Gemma4ClippableLinear(d, h, bias: false, useClipping: useClip)
        self._upProj.wrappedValue = Gemma4ClippableLinear(d, h, bias: false, useClipping: useClip)
        self._downProj.wrappedValue = Gemma4ClippableLinear(h, d, bias: false, useClipping: useClip)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return downProj(geluApproximate(gateProj(x)) * upProj(x))
    }
}

private class Gemma4VisionLayer: Module {
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm: RMSNorm
    @ModuleInfo(key: "self_attn") var selfAttention: Gemma4VisionAttention
    @ModuleInfo var mlp: Gemma4VisionMLP

    init(_ config: Gemma4VisionConfiguration) {
        let d = config.effectiveHiddenSize
        let eps = config.effectiveRmsNormEps
        self._inputLayerNorm.wrappedValue = RMSNorm(dimensions: d, eps: eps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: d, eps: eps)
        self._preFeedforwardLayerNorm.wrappedValue = RMSNorm(dimensions: d, eps: eps)
        self._postFeedforwardLayerNorm.wrappedValue = RMSNorm(dimensions: d, eps: eps)
        self._selfAttention.wrappedValue = Gemma4VisionAttention(config)
        self.mlp = Gemma4VisionMLP(config)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, patchPositions: MLXArray) -> MLXArray {
        // Attention branch: pre-norm, attend, post-norm, residual add
        var residual = x
        var h = inputLayerNorm(x)
        h = selfAttention(h, patchPositions: patchPositions)
        h = postAttentionLayerNorm(h)
        h = residual + h

        // MLP branch
        residual = h
        h = preFeedforwardLayerNorm(h)
        h = mlp(h)
        h = postFeedforwardLayerNorm(h)
        h = residual + h

        return h
    }
}

private class Gemma4VisionEncoder: Module {
    @ModuleInfo var layers: [Gemma4VisionLayer]

    init(_ config: Gemma4VisionConfiguration) {
        self._layers.wrappedValue = (0 ..< config.effectiveNumHiddenLayers).map { _ in Gemma4VisionLayer(config) }
        super.init()
    }

    func callAsFunction(_ x: MLXArray, patchPositions: MLXArray) -> MLXArray {
        var h = x
        for layer in layers {
            h = layer(h, patchPositions: patchPositions)
        }
        return h
    }
}

/// Grid-based average pooling. Inputs: [B, pH*pW, D] and original grid dims (pH, pW).
/// Reshapes to [B, pH, pW, D], runs 2D average pool with kernel=stride=k, flattens back
/// to [B, (pH/k)*(pW/k), D]. Equivalent to Blaizzy's position-aware bin pooling when
/// the patch grid is rectangular and unpadded (which our processor guarantees by
/// aspect-ratio-preserving resize to a multiple of `patch_size * pooling_kernel_size`).
private class Gemma4VisionPooler {
    let kernelSize: Int

    init(_ config: Gemma4VisionConfiguration) {
        self.kernelSize = config.effectivePoolingKernelSize
    }

    func callAsFunction(_ x: MLXArray, pH: Int, pW: Int) -> MLXArray {
        let B = x.dim(0)
        let D = x.dim(2)
        let k = kernelSize

        // [B, pH*pW, D] -> [B, pH, pW, D]
        var h = x.reshaped(B, pH, pW, D)

        // Group by k x k tiles: [B, pH/k, k, pW/k, k, D]
        let outH = pH / k
        let outW = pW / k
        h = h.reshaped(B, outH, k, outW, k, D)
        // Mean over the two k-axes (2 and 4)
        h = h.mean(axes: [2, 4])  // -> [B, outH, outW, D]
        // Flatten spatial: [B, outH*outW, D]
        h = h.reshaped(B, outH * outW, D)
        return h
    }
}

/// Top-level vision model: patch_embedder + encoder + pooler.
private class Gemma4VisionModel: Module {
    @ModuleInfo(key: "patch_embedder") var patchEmbedder: Gemma4VisionPatchEmbedder
    @ModuleInfo var encoder: Gemma4VisionEncoder

    private let pooler: Gemma4VisionPooler
    private let patchSize: Int

    init(_ config: Gemma4VisionConfiguration) {
        self._patchEmbedder.wrappedValue = Gemma4VisionPatchEmbedder(config)
        self._encoder.wrappedValue = Gemma4VisionEncoder(config)
        self.pooler = Gemma4VisionPooler(config)
        self.patchSize = config.effectivePatchSize
        super.init()
    }

    /// Returns soft tokens [B, outH*outW, hiddenSize] where
    /// outH = (H / patch_size) / pooling_kernel_size and similarly for outW.
    func callAsFunction(_ pixelValues: MLXArray) -> MLXArray {
        let B = pixelValues.dim(0)
        let H = pixelValues.dim(2)
        let W = pixelValues.dim(3)
        let pH = H / patchSize
        let pW = W / patchSize

        // Build patch positions [B, pH*pW, 2]: position[i] = (col, row)
        let numPatches = pH * pW
        let idx = MLXArray(Array(0 ..< numPatches).map { Int32($0) })
        let pwArr = Int32(pW)
        // Swift int-div via explicit cast; broadcast of scalars with MLX.
        let colIdx = idx % MLXArray(pwArr)       // col (x)
        let rowIdx = idx / MLXArray(pwArr)       // row (y)
        // [numPatches, 2]  -> stack along axis -1
        let pos2D = stacked([colIdx, rowIdx], axis: -1)  // [numPatches, 2]
        var patchPositions = expandedDimensions(pos2D, axis: 0)  // [1, numPatches, 2]
        if B > 1 {
            patchPositions = repeated(patchPositions, count: B, axis: 0)
        }

        var h = patchEmbedder(pixelValues, patchPositions: patchPositions)
        h = encoder(h, patchPositions: patchPositions)
        h = pooler(h, pH: pH, pW: pW)
        return h
    }
}

// MARK: - Multimodal projector

/// Projects vision soft tokens from the vision-encoder hidden space (768 dim) into the
/// language-model embedding space (e.g. 1536 dim for E2B).
private class Gemma4MultimodalEmbedder: Module {
    @ModuleInfo(key: "embedding_projection") var embeddingProjection: Linear
    @ModuleInfo(key: "embedding_pre_projection_norm") var embeddingPreProjectionNorm: RMSNormNoScale

    init(visionHiddenSize: Int, textHiddenSize: Int, eps: Float = 1e-6) {
        self._embeddingProjection.wrappedValue = Linear(visionHiddenSize, textHiddenSize, bias: false)
        self._embeddingPreProjectionNorm.wrappedValue = RMSNormNoScale(eps: eps)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return embeddingProjection(embeddingPreProjectionNorm(x))
    }
}

// MARK: - Top-level VLM

public class Gemma4VLMModel: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "vision_tower") private var visionTower: Gemma4VisionModel
    @ModuleInfo(key: "language_model") private var languageModel: Gemma4TextModel
    @ModuleInfo(key: "embed_vision") private var embedVision: Gemma4MultimodalEmbedder

    public let config: Gemma4VLMConfiguration

    public var vocabularySize: Int { config.textConfiguration.vocabularySize }
    public var kvHeads: [Int] {
        Array(repeating: config.textConfiguration.kvHeads, count: config.textConfiguration.hiddenLayers)
    }

    public init(_ config: Gemma4VLMConfiguration) {
        self.config = config
        self._visionTower.wrappedValue = Gemma4VisionModel(config.visionConfiguration)
        self._languageModel.wrappedValue = Gemma4TextModel(config.textConfiguration)
        self._embedVision.wrappedValue = Gemma4MultimodalEmbedder(
            visionHiddenSize: config.visionConfiguration.effectiveHiddenSize,
            textHiddenSize: config.textConfiguration.hiddenSize,
            eps: config.visionConfiguration.effectiveRmsNormEps
        )
        super.init()
    }

    public func newCache(parameters: GenerateParameters?) -> [any KVCache] {
        return languageModel.newCache(parameters: parameters)
    }

    /// Fuse vision features into text embeddings at image-token positions.
    private func getInputEmbeddings(
        inputIds: MLXArray,
        pixelValues: MLXArray?
    ) -> MLXArray {
        // Main token embeddings from language model's embed layer
        let mainEmbeds = languageModel.model.embedTokens(inputIds)

        guard let pixels = pixelValues else {
            return mainEmbeds
        }

        // Run vision tower: [B, numSoftTokens, visionHidden]
        var imageFeatures = visionTower(pixels)
        imageFeatures = embedVision(imageFeatures)
        imageFeatures = imageFeatures.asType(mainEmbeds.dtype)

        // Scatter image features into positions where input_ids == image_token_id.
        let imageTokenIdArray = MLXArray(Int32(config.effectiveImageTokenId))
        let imageMask = MLX.equal(inputIds, imageTokenIdArray)  // [B, L] bool
        var maskExpanded = expandedDimensions(imageMask, axis: -1)
        maskExpanded = MLX.broadcast(maskExpanded, to: mainEmbeds.shape)

        // Flatten both and do a bool-mask scatter.
        let shape = mainEmbeds.shape
        let flatMask = maskExpanded.flattened()
        var flatEmbeds = mainEmbeds.flattened()
        let flatImage = imageFeatures.flattened()

        // Compute positions where mask is True; scatter flatImage[0..k] into those positions.
        let maskInts = flatMask.asType(.int32)
        let cumsum = MLX.cumsum(maskInts, axis: 0) - 1
        // Only valid for positions where mask is True. We gather from flatImage by cumsum index.
        // For positions where mask is False, we keep flatEmbeds value via MLX.where.
        let gathered = flatImage[cumsum % flatImage.shape[0]]
        flatEmbeds = MLX.where(flatMask, gathered, flatEmbeds)

        return flatEmbeds.reshaped(shape)
    }

    /// Build per-layer-input (PLE) token IDs: image positions are mapped to pad token ID
    /// so PLE lookup returns a neutral signal (per HF blog guidance).
    private func buildPLETokenIds(_ inputIds: MLXArray) -> MLXArray {
        let imageId = MLXArray(Int32(config.effectiveImageTokenId))
        let padId = MLXArray(Int32(config.effectivePadTokenId))
        let audioIdInt = Int32(config.audioTokenId ?? -1)
        let videoIdInt = Int32(config.videoTokenId ?? -1)

        var ids = MLX.where(MLX.equal(inputIds, imageId), padId, inputIds)
        if audioIdInt >= 0 {
            ids = MLX.where(MLX.equal(ids, MLXArray(audioIdInt)), padId, ids)
        }
        if videoIdInt >= 0 {
            ids = MLX.where(MLX.equal(ids, MLXArray(videoIdInt)), padId, ids)
        }
        return ids
    }

    public func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        // Text-only fast path: no image => delegate to language model directly.
        guard let pixels = input.image?.pixels else {
            guard input.text.tokens.shape[0] > 0 else {
                return .tokens(.init(tokens: MLXArray(Int32(0))[0 ..< 0]))
            }
            return .tokens(input.text)
        }

        // Multimodal path: compute embeddings + PLE, then call language model directly.
        let inputIds = input.text.tokens
        let inputsEmbeds = getInputEmbeddings(inputIds: inputIds, pixelValues: pixels)
        let pleIds = buildPLETokenIds(inputIds)

        let kvCache: [KVCache] = cache.compactMap { $0 as? KVCache }

        let logits = languageModel.callWithEmbeddings(
            inputIds: pleIds,
            inputEmbeddings: inputsEmbeds,
            perLayerInputs: nil,  // Language model computes PLE internally from pleIds
            cache: kvCache
        )

        return .logits(.init(logits: logits))
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        // Text-only per-token step during generation.
        return languageModel(inputs, cache: cache)
    }

    /// Sanitize weights:
    /// 1. Drop audio_tower.* and embed_audio.* (Phase 3: audio deferred)
    /// 2. Partition remaining into language_model.* vs vision_tower.* / embed_vision.*
    /// 3. Delegate language_model partition to Gemma4TextModel.sanitize (handles MoE, prefixes)
    /// 4. Re-prefix with "language_model." for our nested structure
    /// 5. Keep vision_tower and embed_vision weights as-is (ClippableLinear nesting handled by module structure)
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var languageModelWeights: [String: MLXArray] = [:]
        var others: [String: MLXArray] = [:]

        for (k, v) in weights {
            if k.hasPrefix("audio_tower.") || k.hasPrefix("embed_audio.") {
                // Phase 3: audio weights dropped (~300 MB GPU savings)
                continue
            }
            if k.hasPrefix("language_model.") {
                languageModelWeights[k] = v
            } else {
                others[k] = v
            }
        }

        // Gemma4TextModel.sanitize strips the language_model. prefix and returns unprefixed keys.
        let cleanedLM = languageModel.sanitize(weights: languageModelWeights)

        var merged: [String: MLXArray] = [:]
        for (k, v) in cleanedLM {
            // Re-apply the language_model. prefix for our nested submodule hierarchy
            merged["language_model.\(k)"] = v
        }

        // Vision tower / embed_vision keep their original prefixes. ClippableLinear's nested
        // `linear.weight`/`linear.scales`/`linear.biases` layout comes straight from the
        // checkpoint so we pass those through unchanged.
        for (k, v) in others {
            merged[k] = v
        }

        return merged
    }
}

// MARK: - LoRA conformance (delegates to language model)

extension Gemma4VLMModel: LoRAModel {
    public var loraLayers: [Module] { languageModel.loraLayers }
}

// MARK: - Processor

public struct Gemma4Processor: UserInputProcessor {
    private let config: Gemma4ProcessorConfiguration
    private let tokenizer: any Tokenizer

    // Cached token IDs resolved at init time
    private let imageTokenString: String
    private let boiTokenString: String
    private let eoiTokenString: String

    public init(_ config: Gemma4ProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
        // Gemma 4 uses these literal strings (see tokenizer_config.json):
        self.imageTokenString = "<|image|>"
        self.boiTokenString = "<|image>"
        self.eoiTokenString = "<image|>"
    }

    /// Aspect-ratio-preserving resize to target multiple of (patch_size * pooling_kernel_size).
    /// Produces (processedImage, patchesHeight, patchesWidth).
    private func resizeTarget(imageWidth W: CGFloat, imageHeight H: CGFloat) -> (targetW: Int, targetH: Int) {
        let maxSoftTokens = config.effectiveMaxSoftTokens
        let poolingK = config.effectivePoolingKernelSize
        let patchSize = config.effectivePatchSize
        let maxPatches = maxSoftTokens * poolingK * poolingK
        let targetPixels = Double(maxPatches * patchSize * patchSize)

        let factor = sqrt(targetPixels / Double(W * H))
        let sideMult = Double(poolingK * patchSize)
        var targetW = Int(floor(factor * Double(W) / sideMult)) * Int(sideMult)
        var targetH = Int(floor(factor * Double(H) / sideMult)) * Int(sideMult)
        // Ensure minimum of one macro-patch per axis.
        targetW = max(Int(sideMult), targetW)
        targetH = max(Int(sideMult), targetH)
        return (targetW, targetH)
    }

    /// Preprocess a batch of images. Returns (pixel_values as [B, C, H, W], numSoftTokensPerImage).
    public func preprocess(images: [CIImage], processing: UserInput.Processing?) throws -> (MLXArray, [Int], THW) {
        var userProcessing = processing ?? UserInput.Processing()

        var processedArrays: [MLXArray] = []
        var numSoftTokensPerImage: [Int] = []
        var lastTargetH = 0
        var lastTargetW = 0

        for image in images {
            let extent = image.extent
            let (targetW, targetH) = resizeTarget(imageWidth: extent.width, imageHeight: extent.height)
            userProcessing.resize = CGSize(width: targetW, height: targetH)

            let processedImage = MediaProcessing.apply(image, processing: userProcessing)
            let srgbImage = MediaProcessing.inSRGBToneCurveSpace(processedImage)
            let resizedImage = MediaProcessing.resampleBicubic(srgbImage, to: CGSize(width: targetW, height: targetH))

            // Gemma 4 does NOT apply (x - mean)/std. It only does x * (1/255), and the
            // vision encoder itself re-centers with 2 * (x - 0.5). Skip normalize; apply
            // a zero/one mean/std via MediaProcessing to get a [0,1] scaled tensor.
            let zeroMean: (CGFloat, CGFloat, CGFloat) = (0.0, 0.0, 0.0)
            let oneStd: (CGFloat, CGFloat, CGFloat) = (1.0, 1.0, 1.0)
            let normalizedImage = MediaProcessing.normalize(
                resizedImage, mean: zeroMean, std: oneStd)

            let array = MediaProcessing.asMLXArray(normalizedImage)
            processedArrays.append(array)

            // Soft token count for this image
            let patchSize = config.effectivePatchSize
            let poolingK = config.effectivePoolingKernelSize
            let pH = targetH / patchSize
            let pW = targetW / patchSize
            let softTokens = (pH * pW) / (poolingK * poolingK)
            numSoftTokensPerImage.append(softTokens)

            lastTargetH = targetH
            lastTargetW = targetW
        }

        let pixelValues = concatenated(processedArrays)
        // THW for framing; we report the final image size (all images in Phase 3 are single-image).
        return (pixelValues, numSoftTokensPerImage, THW(images.count, lastTargetH, lastTargetW))
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        // Generate chat template; Gemma 4's chat template emits "<|image|>" placeholders
        // for image content items. The processor must then replace each placeholder with
        // boi + image*N + eoi before finalizing the token ID array.
        let messages = Qwen2VLMessageGenerator().generate(from: input)

        // First pass: apply chat template to get raw token IDs (with placeholder image token).
        var promptTokens = try tokenizer.applyChatTemplate(
            messages: messages, tools: input.tools,
            additionalContext: input.additionalContext
        )

        // Resolve special token IDs once.
        let imageTokenId = tokenizer.convertTokenToId(imageTokenString) ?? 258880
        let boiTokenId = tokenizer.convertTokenToId(boiTokenString) ?? 255999
        let eoiTokenId = tokenizer.convertTokenToId(eoiTokenString) ?? 258882

        // Process images if any.
        var processedImage: LMInput.ProcessedImage?
        if !input.images.isEmpty {
            let ciImages = try input.images.map { try $0.asCIImage() }
            let (pixelValues, softTokens, thw) = try preprocess(images: ciImages, processing: input.processing)

            processedImage = LMInput.ProcessedImage(pixels: pixelValues, frames: [thw])

            // Expand each <|image|> placeholder into boi + N × image_token + eoi.
            var expanded: [Int] = []
            expanded.reserveCapacity(promptTokens.count + softTokens.reduce(0, +) + softTokens.count * 2)

            var imageIdx = 0
            for tok in promptTokens {
                if tok == imageTokenId {
                    if imageIdx >= softTokens.count {
                        throw Gemma4VLMError.tokenExpansionFailed(
                            "More image placeholders in chat template than processed images (idx=\(imageIdx), count=\(softTokens.count))")
                    }
                    let n = softTokens[imageIdx]
                    expanded.append(boiTokenId)
                    for _ in 0 ..< n {
                        expanded.append(imageTokenId)
                    }
                    expanded.append(eoiTokenId)
                    imageIdx += 1
                } else {
                    expanded.append(tok)
                }
            }
            // If there were more images than placeholders (unlikely), append the extras at the end.
            while imageIdx < softTokens.count {
                let n = softTokens[imageIdx]
                expanded.append(boiTokenId)
                for _ in 0 ..< n {
                    expanded.append(imageTokenId)
                }
                expanded.append(eoiTokenId)
                imageIdx += 1
            }

            promptTokens = expanded
        }

        let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
        let mask = ones(like: promptArray).asType(.int8)
        return LMInput(
            text: .init(tokens: promptArray, mask: mask),
            image: processedImage
        )
    }
}
