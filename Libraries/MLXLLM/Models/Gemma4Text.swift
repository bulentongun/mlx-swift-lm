//
//  Gemma4Text.swift
//  mlx-swift-lm
//
//  Ported from https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/gemma4/language.py
//  Supports E2B (2.3B effective), E4B (4.5B effective), 26B-A4B (MoE), 31B (dense)
//  Key features: PLE (Per-Layer Embeddings), Shared KV Cache, K-eq-V, Dual RoPE, MoE, Logit Softcapping
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

public struct Gemma4TextConfiguration: Codable, Sendable {
    let modelType: String
    let hiddenSize: Int
    let hiddenLayers: Int
    let intermediateSize: Int
    let attentionHeads: Int
    let headDim: Int
    let globalHeadDim: Int?
    let rmsNormEps: Float
    let vocabularySize: Int
    let kvHeads: Int
    let globalKVHeads: Int?
    let ropeTraditional: Bool
    let slidingWindow: Int
    let slidingWindowPattern: Int
    let maxPositionEmbeddings: Int
    let hiddenActivation: String?

    // PLE (Per-Layer Embeddings) — E2B/E4B only
    let hiddenSizePerLayerInput: Int
    let vocabSizePerLayerInput: Int?

    // Shared KV Cache
    let numKVSharedLayers: Int

    // K-eq-V (26B/31B)
    let attentionKEqV: Bool

    // MoE (26B only)
    let enableMoeBlock: Bool
    let numExperts: Int?
    let topKExperts: Int?
    let moeIntermediateSize: Int?
    let useDoubleWideMLP: Bool

    // Logit softcapping
    let finalLogitSoftcapping: Float?

    // RoPE parameters (dual)
    let ropeParameters: [String: [String: StringOrNumber]]?

    // Layer types
    let layerTypes: [String]?

    // Partial rotary
    let partialRotaryFactor: Float?
    let globalPartialRotaryFactor: Float?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case headDim = "head_dim"
        case globalHeadDim = "global_head_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case globalKVHeads = "num_global_key_value_heads"
        case ropeTraditional = "rope_traditional"
        case slidingWindow = "sliding_window"
        case slidingWindowPattern = "sliding_window_pattern"
        case maxPositionEmbeddings = "max_position_embeddings"
        case hiddenActivation = "hidden_activation"
        case hiddenSizePerLayerInput = "hidden_size_per_layer_input"
        case vocabSizePerLayerInput = "vocab_size_per_layer_input"
        case numKVSharedLayers = "num_kv_shared_layers"
        case attentionKEqV = "attention_k_eq_v"
        case enableMoeBlock = "enable_moe_block"
        case numExperts = "num_experts"
        case topKExperts = "top_k_experts"
        case moeIntermediateSize = "moe_intermediate_size"
        case useDoubleWideMLP = "use_double_wide_mlp"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case ropeParameters = "rope_parameters"
        case layerTypes = "layer_types"
        case partialRotaryFactor = "partial_rotary_factor"
        case globalPartialRotaryFactor = "global_partial_rotary_factor"
    }

    enum VLMCodingKeys: String, CodingKey {
        case textConfig = "text_config"
    }

    public init(from decoder: Decoder) throws {
        let nestedContainer = try decoder.container(keyedBy: VLMCodingKeys.self)
        let container =
            if nestedContainer.contains(.textConfig) {
                try nestedContainer.nestedContainer(keyedBy: CodingKeys.self, forKey: .textConfig)
            } else {
                try decoder.container(keyedBy: CodingKeys.self)
            }

        modelType = try container.decode(String.self, forKey: .modelType)
        hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1536
        hiddenLayers = try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 35
        intermediateSize = try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 6144
        attentionHeads = try container.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 8
        headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? 256
        globalHeadDim = try container.decodeIfPresent(Int.self, forKey: .globalHeadDim)
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        vocabularySize = try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 262144
        kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 1
        globalKVHeads = try container.decodeIfPresent(Int.self, forKey: .globalKVHeads)
        ropeTraditional = try container.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? false
        slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 512
        slidingWindowPattern = try container.decodeIfPresent(Int.self, forKey: .slidingWindowPattern) ?? 5
        maxPositionEmbeddings = try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131072
        hiddenActivation = try container.decodeIfPresent(String.self, forKey: .hiddenActivation)
        hiddenSizePerLayerInput = try container.decodeIfPresent(Int.self, forKey: .hiddenSizePerLayerInput) ?? 0
        vocabSizePerLayerInput = try container.decodeIfPresent(Int.self, forKey: .vocabSizePerLayerInput)
        numKVSharedLayers = try container.decodeIfPresent(Int.self, forKey: .numKVSharedLayers) ?? 0
        attentionKEqV = try container.decodeIfPresent(Bool.self, forKey: .attentionKEqV) ?? false
        enableMoeBlock = try container.decodeIfPresent(Bool.self, forKey: .enableMoeBlock) ?? false
        numExperts = try container.decodeIfPresent(Int.self, forKey: .numExperts)
        topKExperts = try container.decodeIfPresent(Int.self, forKey: .topKExperts)
        moeIntermediateSize = try container.decodeIfPresent(Int.self, forKey: .moeIntermediateSize)
        useDoubleWideMLP = try container.decodeIfPresent(Bool.self, forKey: .useDoubleWideMLP) ?? false
        finalLogitSoftcapping = try container.decodeIfPresent(Float.self, forKey: .finalLogitSoftcapping)
        ropeParameters = try container.decodeIfPresent([String: [String: StringOrNumber]].self, forKey: .ropeParameters)
        layerTypes = try container.decodeIfPresent([String].self, forKey: .layerTypes)
        partialRotaryFactor = try container.decodeIfPresent(Float.self, forKey: .partialRotaryFactor)
        globalPartialRotaryFactor = try container.decodeIfPresent(Float.self, forKey: .globalPartialRotaryFactor)
    }

    /// Compute layer types if not explicitly provided in config
    func resolvedLayerTypes() -> [String] {
        if let types = layerTypes, !types.isEmpty {
            return types
        }
        // Default pattern: slidingWindowPattern-1 sliding + 1 full, repeated
        return (0 ..< hiddenLayers).map { i in
            ((i + 1) % slidingWindowPattern == 0) ? "full_attention" : "sliding_attention"
        }
    }

    /// First KV-shared layer index
    var firstKVSharedLayerIdx: Int {
        hiddenLayers - numKVSharedLayers
    }
}

// MARK: - RMSNorm Variants

/// RMSNorm without learnable scale (Gemma4-specific)
class Gemma4RMSNormNoScale: Module, UnaryLayer {
    let eps: Float

    init(dimensions: Int, eps: Float = 1e-6) {
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // rmsNorm with nil weight = no scale
        let variance = (x * x).mean(axis: -1, keepDims: true)
        return x * MLX.rsqrt(variance + eps)
    }
}

/// RMSNorm with zero-shift (weight used directly, no +1 offset like standard Gemma)
class Gemma4RMSNormZeroShift: Module, UnaryLayer {
    let weight: MLXArray
    let eps: Float

    init(dimensions: Int, eps: Float = 1e-6) {
        self.weight = MLXArray.ones([dimensions])
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Standard Gemma RMSNorm uses (1.0 + weight), Gemma4 uses weight directly
        return MLXFast.rmsNorm(x, weight: self.weight, eps: self.eps)
    }
}

// MARK: - Gemma4ScaledLinear

/// Linear layer with output scaling (used for PLE model projection)
class Gemma4ScaledLinear: Module {
    let weight: MLXArray
    let scalar: Float

    init(inputDims: Int, outputDims: Int, scalar: Float) {
        self.weight = MLXArray.zeros([outputDims, inputDims])
        self.scalar = scalar
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return matmul(x, weight.T) * scalar
    }
}

// MARK: - MLP

class Gemma4MLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        self._gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
        self._upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return downProj(geluApproximate(gateProj(x)) * upProj(x))
    }
}

// MARK: - MoE Router (26B only)

class Gemma4Router: Module {
    @ModuleInfo var norm: Gemma4RMSNormNoScale
    @ModuleInfo var proj: Linear
    let scale: MLXArray
    let perExpertScale: MLXArray
    let rootSize: Float
    let topK: Int

    init(_ config: Gemma4TextConfiguration) {
        let nExperts = config.numExperts ?? 128
        self.topK = config.topKExperts ?? 8
        self.rootSize = pow(Float(config.hiddenSize), -0.5)
        self._norm.wrappedValue = Gemma4RMSNormNoScale(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._proj.wrappedValue = Linear(config.hiddenSize, nExperts, bias: false)
        self.scale = MLXArray.ones([config.hiddenSize])
        self.perExpertScale = MLXArray.ones([nExperts])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        var h = norm(x)
        h = h * rootSize
        h = h * scale

        let expertScores = proj(h)
        let routerProbs = softmax(expertScores, axis: -1)

        let topKIndices = argSort(-expertScores, axis: -1)[.ellipsis, ..<topK]
        var topKWeights = MLX.takeAlong(routerProbs, topKIndices, axis: -1)
        topKWeights = topKWeights / topKWeights.sum(axis: -1, keepDims: true)
        topKWeights = topKWeights * perExpertScale[topKIndices]

        return (topKIndices, topKWeights)
    }
}

// MARK: - MoE Experts (26B only)

class Gemma4Experts: Module {
    @ModuleInfo(key: "switch_glu") var switchGLU: SwitchGLU

    init(_ config: Gemma4TextConfiguration) {
        let nExperts = config.numExperts ?? 128
        let moeSize = config.moeIntermediateSize ?? config.intermediateSize
        self._switchGLU.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: moeSize,
            numExperts: nExperts,
            activation: geluApproximate,
            bias: false
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray, indices: MLXArray, weights: MLXArray) -> MLXArray {
        let (B, S, H) = (x.dim(0), x.dim(1), x.dim(2))
        let K = indices.dim(-1)

        let xFlat = x.reshaped(B * S, H)
        let indicesFlat = indices.reshaped(B * S, K)

        let expertOut = switchGLU(xFlat, indicesFlat)

        let w = weights.reshaped(B * S, K)[.ellipsis, .newAxis]
        return (expertOut * w).sum(axis: -2).reshaped(B, S, H)
    }
}

// MARK: - Attention

class Gemma4Attention: Module {
    let config: Gemma4TextConfiguration
    let layerIdx: Int
    let layerType: String
    let isSliding: Bool
    let nHeads: Int
    let nKVHeads: Int
    let effectiveHeadDim: Int
    let scale: Float
    let useKEqV: Bool
    let isKVSharedLayer: Bool

    @ModuleInfo(key: "q_proj") var queryProj: Linear
    @ModuleInfo(key: "k_proj") var keyProj: Linear
    @ModuleInfo(key: "v_proj") var valueProj: Linear?
    @ModuleInfo(key: "o_proj") var outputProj: Linear

    @ModuleInfo(key: "q_norm") var queryNorm: Gemma.RMSNorm
    @ModuleInfo(key: "k_norm") var keyNorm: Gemma.RMSNorm
    @ModuleInfo(key: "v_norm") var valueNorm: Gemma4RMSNormNoScale

    @ModuleInfo var rope: RoPELayer

    init(_ config: Gemma4TextConfiguration, layerIdx: Int) {
        self.config = config
        self.layerIdx = layerIdx
        let resolvedTypes = config.resolvedLayerTypes()
        self.layerType = resolvedTypes[layerIdx]
        self.isSliding = layerType == "sliding_attention"

        // Head dim: global layers may use different head dim
        if !isSliding, let ghd = config.globalHeadDim, ghd > 0 {
            self.effectiveHeadDim = ghd
        } else {
            self.effectiveHeadDim = config.headDim
        }

        let dim = config.hiddenSize
        self.nHeads = config.attentionHeads

        // K-eq-V: full attention layers on 26B/31B
        self.useKEqV = config.attentionKEqV && !isSliding
        if useKEqV, let gkv = config.globalKVHeads {
            self.nKVHeads = gkv
        } else {
            self.nKVHeads = config.kvHeads
        }

        // Scale = 1.0 (Gemma4 uses norm-based scaling, not 1/sqrt(d))
        self.scale = 1.0

        // KV sharing
        self.isKVSharedLayer = layerIdx >= config.firstKVSharedLayerIdx && config.firstKVSharedLayerIdx > 0

        self._queryProj.wrappedValue = Linear(dim, nHeads * effectiveHeadDim, bias: false)
        self._keyProj.wrappedValue = Linear(dim, nKVHeads * effectiveHeadDim, bias: false)
        if !useKEqV {
            self._valueProj.wrappedValue = Linear(dim, nKVHeads * effectiveHeadDim, bias: false)
        }
        self._outputProj.wrappedValue = Linear(nHeads * effectiveHeadDim, dim, bias: false)

        self._queryNorm.wrappedValue = Gemma.RMSNorm(dimensions: effectiveHeadDim, eps: config.rmsNormEps)
        self._keyNorm.wrappedValue = Gemma.RMSNorm(dimensions: effectiveHeadDim, eps: config.rmsNormEps)
        self._valueNorm.wrappedValue = Gemma4RMSNormNoScale(dimensions: effectiveHeadDim, eps: config.rmsNormEps)

        // Dual RoPE: sliding uses local theta, full uses proportional theta
        let ropeParams: [String: StringOrNumber]?
        if isSliding {
            ropeParams = config.ropeParameters?["sliding_attention"]?.mapValues { $0 }
        } else {
            ropeParams = config.ropeParameters?["full_attention"]?.mapValues { $0 }
        }
        let ropeTheta: Float = ropeParams?["rope_theta"]?.asFloat() ?? (isSliding ? 10_000.0 : 1_000_000.0)

        self.rope = initializeRope(
            dims: effectiveHeadDim,
            base: ropeTheta,
            traditional: config.ropeTraditional,
            scalingConfig: ropeParams,
            maxPositionEmbeddings: config.maxPositionEmbeddings
        )

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache? = nil
    ) -> MLXArray {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

        var queries = queryProj(x).reshaped(B, L, nHeads, effectiveHeadDim)
        queries = queryNorm(queries)

        var offset = 0

        // Shared KV: reuse cache from earlier layer
        var keys: MLXArray
        var values: MLXArray

        if isKVSharedLayer, let cache {
            // Read directly from shared cache (already populated by earlier layer)
            let state = cache.state
            keys = state[0]
            values = state[1]
            offset = cache.offset

            // Still need to compute and apply RoPE to queries
            queries = queries.transposed(0, 2, 1, 3)
            queries = rope(queries, offset: offset)
        } else {
            if let cache {
                offset = cache.offset
            }

            keys = keyProj(x).reshaped(B, L, nKVHeads, effectiveHeadDim)

            // K-eq-V: values = raw keys (before normalization)
            if useKEqV {
                values = keys
            } else if let vp = valueProj {
                values = vp(x).reshaped(B, L, nKVHeads, effectiveHeadDim)
            } else {
                values = keys
            }

            keys = keyNorm(keys)
            values = valueNorm(values)

            keys = keys.transposed(0, 2, 1, 3)
            values = values.transposed(0, 2, 1, 3)

            keys = rope(keys, offset: offset)

            if let cache {
                let (ck, cv) = cache.update(keys: keys, values: values)
                keys = ck
                values = cv
            }

            queries = queries.transposed(0, 2, 1, 3)
            queries = rope(queries, offset: offset)
        }

        let output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return outputProj(output)
    }
}

// MARK: - Decoder Layer

class Gemma4DecoderLayer: Module {
    let config: Gemma4TextConfiguration
    let layerIdx: Int
    let layerType: String
    let enableMoE: Bool
    let hasPLE: Bool

    @ModuleInfo(key: "self_attn") var selfAttention: Gemma4Attention
    @ModuleInfo var mlp: Gemma4MLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: Gemma.RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: Gemma.RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm: Gemma.RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm: Gemma.RMSNorm

    // MoE (26B)
    @ModuleInfo var router: Gemma4Router?
    @ModuleInfo var experts: Gemma4Experts?
    @ModuleInfo(key: "pre_feedforward_layernorm_2") var preFeedforwardLayerNorm2: Gemma.RMSNorm?
    @ModuleInfo(key: "post_feedforward_layernorm_1") var postFeedforwardLayerNorm1: Gemma.RMSNorm?
    @ModuleInfo(key: "post_feedforward_layernorm_2") var postFeedforwardLayerNorm2: Gemma.RMSNorm?

    // PLE gating
    @ModuleInfo(key: "per_layer_input_gate") var perLayerInputGate: Linear?
    @ModuleInfo(key: "per_layer_projection") var perLayerProjection: Linear?
    @ModuleInfo(key: "post_per_layer_input_norm") var postPerLayerInputNorm: Gemma.RMSNorm?

    // Layer scalar (loaded from weights)
    @ParameterInfo(key: "layer_scalar") var layerScalar: MLXArray

    init(_ config: Gemma4TextConfiguration, layerIdx: Int) {
        self.config = config
        self.layerIdx = layerIdx
        let resolvedTypes = config.resolvedLayerTypes()
        self.layerType = resolvedTypes[layerIdx]
        self.enableMoE = config.enableMoeBlock
        self.hasPLE = config.hiddenSizePerLayerInput > 0

        // Attention
        self._selfAttention.wrappedValue = Gemma4Attention(config, layerIdx: layerIdx)

        // MLP — double-wide for KV-shared layers
        let isKVShared = layerIdx >= config.firstKVSharedLayerIdx && config.firstKVSharedLayerIdx > 0
        let effectiveIntermediate = (config.useDoubleWideMLP && isKVShared)
            ? config.intermediateSize * 2
            : config.intermediateSize
        self.mlp = Gemma4MLP(hiddenSize: config.hiddenSize, intermediateSize: effectiveIntermediate)

        // Norms
        self._inputLayerNorm.wrappedValue = Gemma.RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = Gemma.RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._preFeedforwardLayerNorm.wrappedValue = Gemma.RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postFeedforwardLayerNorm.wrappedValue = Gemma.RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        // MoE components (26B only)
        if enableMoE {
            self._router.wrappedValue = Gemma4Router(config)
            self._experts.wrappedValue = Gemma4Experts(config)
            self._preFeedforwardLayerNorm2.wrappedValue = Gemma.RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
            self._postFeedforwardLayerNorm1.wrappedValue = Gemma.RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
            self._postFeedforwardLayerNorm2.wrappedValue = Gemma.RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }

        // PLE gating (E2B/E4B)
        if hasPLE {
            let pleSize = config.hiddenSizePerLayerInput
            self._perLayerInputGate.wrappedValue = Linear(config.hiddenSize, pleSize, bias: false)
            self._perLayerProjection.wrappedValue = Linear(pleSize, config.hiddenSize, bias: false)
            self._postPerLayerInputNorm.wrappedValue = Gemma.RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }

        // Layer scalar (will be loaded from weights, default ones)
        self._layerScalar.wrappedValue = MLXArray.ones([1])

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache? = nil,
        perLayerInput: MLXArray? = nil
    ) -> MLXArray {
        // Self-attention
        var residual = x
        var h = inputLayerNorm(x)
        h = selfAttention(h, mask: mask, cache: cache)
        h = postAttentionLayerNorm(h)
        h = residual + h

        // Feed-forward (dense or MoE)
        residual = h

        if enableMoE, let router, let experts,
           let preNorm2 = preFeedforwardLayerNorm2,
           let postNorm1 = postFeedforwardLayerNorm1,
           let postNorm2 = postFeedforwardLayerNorm2
        {
            // Dense MLP path
            var h1 = preFeedforwardLayerNorm(h)
            h1 = mlp(h1)
            h1 = postNorm1(h1)

            // Expert MoE path
            let (topKIndices, topKWeights) = router(h)
            var h2 = preNorm2(h)
            h2 = experts(h2, indices: topKIndices, weights: topKWeights)
            h2 = postNorm2(h2)

            h = h1 + h2
        } else {
            h = preFeedforwardLayerNorm(h)
            h = mlp(h)
        }

        h = postFeedforwardLayerNorm(h)
        h = residual + h

        // PLE injection
        if hasPLE,
           let gate = perLayerInputGate,
           let proj = perLayerProjection,
           let postNorm = postPerLayerInputNorm,
           let pli = perLayerInput
        {
            residual = h
            var g = gate(h)
            g = geluApproximate(g)
            g = g * pli
            g = proj(g)
            g = postNorm(g)
            h = residual + g
        }

        // Layer scaling
        h = h * layerScalar

        return h
    }
}

// MARK: - Text Model

public class Gemma4Model: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [Gemma4DecoderLayer]
    @ModuleInfo var norm: Gemma.RMSNorm

    // PLE embeddings (E2B/E4B)
    @ModuleInfo(key: "embed_tokens_per_layer") var embedTokensPerLayer: Embedding?
    @ModuleInfo(key: "per_layer_model_projection") var perLayerModelProjection: Gemma4ScaledLinear?
    @ModuleInfo(key: "per_layer_projection_norm") var perLayerProjectionNorm: Gemma4RMSNormZeroShift?

    let config: Gemma4TextConfiguration
    let embedScale: Float
    let hasPLE: Bool
    let pleScale: Float
    let pleEmbedScale: Float

    // KV sharing mapping
    let layerIdxToCacheIdx: [Int]
    let firstKVSharedLayerIdx: Int
    let firstFullCacheIdx: Int
    let firstSlidingCacheIdx: Int

    init(_ config: Gemma4TextConfiguration) {
        self.config = config
        self.embedScale = sqrt(Float(config.hiddenSize))
        self.hasPLE = config.hiddenSizePerLayerInput > 0
        self.pleScale = pow(2.0, -0.5)
        self.pleEmbedScale = sqrt(Float(config.hiddenSizePerLayerInput > 0 ? config.hiddenSizePerLayerInput : 1))
        self.firstKVSharedLayerIdx = config.firstKVSharedLayerIdx

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )

        self._layers.wrappedValue = (0 ..< config.hiddenLayers).map { i in
            Gemma4DecoderLayer(config, layerIdx: i)
        }

        self.norm = Gemma.RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        // PLE embedding table
        if hasPLE {
            let pleVocab = config.vocabSizePerLayerInput ?? config.vocabularySize
            let pleTotalDim = config.hiddenLayers * config.hiddenSizePerLayerInput
            self._embedTokensPerLayer.wrappedValue = Embedding(
                embeddingCount: pleVocab,
                dimensions: pleTotalDim
            )
            self._perLayerModelProjection.wrappedValue = Gemma4ScaledLinear(
                inputDims: config.hiddenSize,
                outputDims: pleTotalDim,
                scalar: pow(Float(config.hiddenSize), -0.5)
            )
            self._perLayerProjectionNorm.wrappedValue = Gemma4RMSNormZeroShift(
                dimensions: config.hiddenSizePerLayerInput,
                eps: config.rmsNormEps
            )
        }

        // Build KV sharing map
        let resolvedTypes = config.resolvedLayerTypes()
        let concreteTypes = Array(resolvedTypes[..<firstKVSharedLayerIdx])

        var mapping = Array(0 ..< firstKVSharedLayerIdx)
        if firstKVSharedLayerIdx < config.hiddenLayers {
            let sharedFullIdx = concreteTypes.lastIndex(of: "full_attention") ?? 0
            let sharedSlidingIdx = concreteTypes.lastIndex(of: "sliding_attention") ?? 0
            for i in firstKVSharedLayerIdx ..< config.hiddenLayers {
                if resolvedTypes[i] == "full_attention" {
                    mapping.append(sharedFullIdx)
                } else {
                    mapping.append(sharedSlidingIdx)
                }
            }
        }
        self.layerIdxToCacheIdx = mapping

        self.firstFullCacheIdx = concreteTypes.firstIndex(of: "full_attention") ?? 0
        self.firstSlidingCacheIdx = concreteTypes.firstIndex(of: "sliding_attention") ?? 0

        super.init()
    }

    /// Get per-layer embeddings from token IDs
    private func getPerLayerInputs(_ inputIds: MLXArray) -> MLXArray {
        guard let embed = embedTokensPerLayer else {
            fatalError("PLE embedding not initialized")
        }
        var result = embed(inputIds)
        result = result * pleEmbedScale
        // Reshape to [B, S, numLayers, pleSize]
        let shape = inputIds.shape
        return result.reshaped(shape[0], shape[1], config.hiddenLayers, config.hiddenSizePerLayerInput)
    }

    /// Project hidden states to per-layer inputs and combine with token PLE
    private func projectPerLayerInputs(
        _ inputsEmbeds: MLXArray,
        perLayerInputs: MLXArray?
    ) -> MLXArray {
        guard let proj = perLayerModelProjection, let normLayer = perLayerProjectionNorm else {
            fatalError("PLE projection not initialized")
        }
        var perLayerProj = proj(inputsEmbeds)
        let shape = inputsEmbeds.shape
        perLayerProj = perLayerProj.reshaped(
            shape[0], shape[1], config.hiddenLayers, config.hiddenSizePerLayerInput
        )
        perLayerProj = normLayer(perLayerProj)

        guard let pli = perLayerInputs else {
            return perLayerProj
        }
        return (perLayerProj + pli) * pleScale
    }

    func callAsFunction(
        _ inputs: MLXArray,
        inputsEmbeds: MLXArray? = nil,
        mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        cache: [KVCache?]? = nil,
        perLayerInputs: MLXArray? = nil
    ) -> MLXArray {
        var h: MLXArray
        if let embeds = inputsEmbeds {
            h = embeds
        } else {
            h = embedTokens(inputs)
            h = h * MLXArray(embedScale, dtype: .bfloat16).asType(h.dtype)
        }

        // PLE: compute per-layer inputs
        // PLE restored after diagnosis — garbage was caused by wrong chat template, not PLE
        let pleEnabled = hasPLE
        var resolvedPLI = perLayerInputs
        if pleEnabled {
            if perLayerInputs == nil {
                resolvedPLI = getPerLayerInputs(inputs)
            }
            resolvedPLI = projectPerLayerInputs(h, perLayerInputs: resolvedPLI)
        }

        // Setup cache
        var layerCache: [KVCache?] = cache ?? Array(repeating: nil, count: firstKVSharedLayerIdx)

        // Compute masks
        let globalMask: MLXFast.ScaledDotProductAttentionMaskMode
        let slidingWindowMask: MLXFast.ScaledDotProductAttentionMaskMode

        if let m = mask {
            globalMask = m
            slidingWindowMask = m
        } else {
            globalMask = createAttentionMask(
                h: h,
                cache: firstFullCacheIdx < layerCache.count ? layerCache[firstFullCacheIdx] : nil
            )
            slidingWindowMask = createAttentionMask(
                h: h,
                cache: firstSlidingCacheIdx < layerCache.count ? layerCache[firstSlidingCacheIdx] : nil,
                windowSize: config.slidingWindow
            )
        }

        // Iterate decoder layers
        let resolvedTypes = config.resolvedLayerTypes()
        for (i, layer) in layers.enumerated() {
            let cacheIdx = layerIdxToCacheIdx[i]
            let c = cacheIdx < layerCache.count ? layerCache[cacheIdx] : nil
            let isGlobal = resolvedTypes[i] == "full_attention"
            let layerMask = isGlobal ? globalMask : slidingWindowMask

            let pli: MLXArray? = resolvedPLI != nil ? resolvedPLI![.ellipsis, i, 0...] : nil

            h = layer(h, mask: layerMask, cache: c, perLayerInput: pli)
        }

        return norm(h)
    }
}

// MARK: - Language Model (Top-level)

public class Gemma4TextModel: Module, LLMModel {
    @ModuleInfo public var model: Gemma4Model
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public let config: Gemma4TextConfiguration
    public var vocabularySize: Int { config.vocabularySize }

    let finalLogitSoftcapping: Float?

    public init(_ config: Gemma4TextConfiguration) {
        self.config = config
        self.finalLogitSoftcapping = config.finalLogitSoftcapping
        self.model = Gemma4Model(config)
        self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var out = model(inputs, cache: cache)
        out = lmHead(out)

        // Logit softcapping
        if let cap = finalLogitSoftcapping {
            out = tanh(out / cap) * cap
        }

        return out
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var processedWeights = weights

        // VLM models have weights under language_model key
        let unflattened = ModuleParameters.unflattened(weights)
        if let lm = unflattened["language_model"] {
            processedWeights = Dictionary(uniqueKeysWithValues: lm.flattened())
        }

        // Filter clipping params (vision/audio only)
        processedWeights = processedWeights.filter { k, _ in
            if k.contains("self_attn.rotary_emb") { return false }
            if k.contains("input_max") || k.contains("input_min")
                || k.contains("output_max") || k.contains("output_min")
            {
                if !k.contains("vision_tower") && !k.contains("audio_tower") {
                    return false
                }
            }
            return true
        }

        // Vocab size truncation
        let expectedVocab = config.vocabularySize
        let keysToCheck = [
            "model.embed_tokens.weight", "model.embed_tokens.scales", "model.embed_tokens.biases",
            "lm_head.weight", "lm_head.scales", "lm_head.biases",
        ]
        for key in keysToCheck {
            if let tensor = processedWeights[key], tensor.dim(0) > expectedVocab {
                processedWeights[key] = tensor[0 ..< expectedVocab]
            }
        }

        // Tie word embeddings
        if processedWeights["lm_head.weight"] == nil {
            ["weight", "scales", "biases"].forEach { key in
                if let embedWeight = processedWeights["model.embed_tokens.\(key)"] {
                    processedWeights["lm_head.\(key)"] = embedWeight
                }
            }
        }

        return processedWeights
    }

    public func newCache(parameters: GenerateParameters? = nil) -> [KVCache] {
        var caches = [KVCache]()
        let resolvedTypes = config.resolvedLayerTypes()
        let numConcreteLayers = model.firstKVSharedLayerIdx

        for i in 0 ..< numConcreteLayers {
            if resolvedTypes[i] == "full_attention" {
                let cache = StandardKVCache()
                cache.step = 1024
                caches.append(cache)
            } else {
                caches.append(
                    RotatingKVCache(maxSize: config.slidingWindow, keep: 0)
                )
            }
        }

        return caches
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], windowSize: Int? = nil
    ) throws -> PrepareResult {
        let promptTokens = input.text.tokens
        guard promptTokens.shape[0] > 0 else {
            let emptyToken = MLXArray(Int32(0))[0 ..< 0]
            return .tokens(.init(tokens: emptyToken))
        }
        return .tokens(input.text)
    }
}

extension Gemma4TextModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
