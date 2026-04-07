//
//  Gemma4Text.swift
//  mlx-swift-lm
//
//  Exact port from ml-explore/mlx-lm/mlx_lm/models/gemma4_text.py
//  Commit reference: f26fddf + c65c27b
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
    let numKVSharedLayers: Int
    let hiddenSizePerLayerInput: Int
    let vocabSizePerLayerInput: Int?
    let ropeTraditional: Bool
    let ropeParameters: [String: [String: StringOrNumber]]?
    let slidingWindow: Int
    let slidingWindowPattern: Int?
    let maxPositionEmbeddings: Int
    let attentionKEqV: Bool
    let finalLogitSoftcapping: Float?
    let useDoubleWideMLP: Bool
    let enableMoeBlock: Bool
    let numExperts: Int?
    let topKExperts: Int?
    let moeIntermediateSize: Int?
    let layerTypes: [String]?
    let tieWordEmbeddings: Bool

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
        case numKVSharedLayers = "num_kv_shared_layers"
        case hiddenSizePerLayerInput = "hidden_size_per_layer_input"
        case vocabSizePerLayerInput = "vocab_size_per_layer_input"
        case ropeTraditional = "rope_traditional"
        case ropeParameters = "rope_parameters"
        case slidingWindow = "sliding_window"
        case slidingWindowPattern = "sliding_window_pattern"
        case maxPositionEmbeddings = "max_position_embeddings"
        case attentionKEqV = "attention_k_eq_v"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case useDoubleWideMLP = "use_double_wide_mlp"
        case enableMoeBlock = "enable_moe_block"
        case numExperts = "num_experts"
        case topKExperts = "top_k_experts"
        case moeIntermediateSize = "moe_intermediate_size"
        case layerTypes = "layer_types"
        case tieWordEmbeddings = "tie_word_embeddings"
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
        numKVSharedLayers = try container.decodeIfPresent(Int.self, forKey: .numKVSharedLayers) ?? 0
        hiddenSizePerLayerInput = try container.decodeIfPresent(Int.self, forKey: .hiddenSizePerLayerInput) ?? 0
        vocabSizePerLayerInput = try container.decodeIfPresent(Int.self, forKey: .vocabSizePerLayerInput)
        ropeTraditional = try container.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? false
        ropeParameters = try container.decodeIfPresent([String: [String: StringOrNumber]].self, forKey: .ropeParameters)
        slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 512
        slidingWindowPattern = try container.decodeIfPresent(Int.self, forKey: .slidingWindowPattern)
        maxPositionEmbeddings = try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131072
        attentionKEqV = try container.decodeIfPresent(Bool.self, forKey: .attentionKEqV) ?? false
        finalLogitSoftcapping = try container.decodeIfPresent(Float.self, forKey: .finalLogitSoftcapping)
        useDoubleWideMLP = try container.decodeIfPresent(Bool.self, forKey: .useDoubleWideMLP) ?? true
        enableMoeBlock = try container.decodeIfPresent(Bool.self, forKey: .enableMoeBlock) ?? false
        numExperts = try container.decodeIfPresent(Int.self, forKey: .numExperts)
        topKExperts = try container.decodeIfPresent(Int.self, forKey: .topKExperts)
        moeIntermediateSize = try container.decodeIfPresent(Int.self, forKey: .moeIntermediateSize)
        layerTypes = try container.decodeIfPresent([String].self, forKey: .layerTypes)
        tieWordEmbeddings = try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
    }

    func resolvedLayerTypes() -> [String] {
        if let types = layerTypes, !types.isEmpty { return types }
        let swp = slidingWindowPattern ?? 5
        let pattern = Array(repeating: "sliding_attention", count: swp - 1) + ["full_attention"]
        return Array((0 ..< hiddenLayers).map { pattern[$0 % pattern.count] })
    }

    var firstKVSharedLayerIdx: Int { hiddenLayers - numKVSharedLayers }
}

// MARK: - RMSNormNoScale

class Gemma4RMSNormNoScale: Module, UnaryLayer {
    let eps: Float
    init(dimensions: Int, eps: Float = 1e-6) {
        self.eps = eps
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let variance = (x * x).mean(axis: -1, keepDims: true)
        return x * MLX.rsqrt(variance + eps)
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
        downProj(geluApproximate(gateProj(x)) * upProj(x))
    }
}

// MARK: - MoE

class Gemma4Router: Module {
    @ModuleInfo var proj: Linear
    let scale: MLXArray
    let perExpertScale: MLXArray
    let rootSize: Float
    let eps: Float
    let topK: Int

    init(_ config: Gemma4TextConfiguration) {
        let nExperts = config.numExperts ?? 128
        self.topK = config.topKExperts ?? 8
        self.eps = config.rmsNormEps
        self.rootSize = pow(Float(config.hiddenSize), -0.5)
        self._proj.wrappedValue = Linear(config.hiddenSize, nExperts, bias: false)
        self.scale = MLXArray.ones([config.hiddenSize])
        self.perExpertScale = MLXArray.ones([nExperts])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let h = MLXFast.rmsNorm(x, weight: scale * rootSize, eps: eps)
        let expertScores = proj(h)
        let topKIndices = argSort(-expertScores, axis: -1)[.ellipsis, ..<topK]
        var topKWeights = MLX.takeAlong(expertScores, topKIndices, axis: -1)
        topKWeights = softmax(topKWeights, axis: -1)
        topKWeights = topKWeights * perExpertScale[topKIndices]
        return (topKIndices, topKWeights)
    }
}

class Gemma4Experts: Module {
    @ModuleInfo(key: "switch_glu") var switchGLU: SwitchGLU

    init(_ config: Gemma4TextConfiguration) {
        let nExperts = config.numExperts ?? 128
        let moeSize = config.moeIntermediateSize ?? config.intermediateSize
        self._switchGLU.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize, hiddenDims: moeSize,
            numExperts: nExperts, activation: geluApproximate, bias: false
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray, indices: MLXArray, weights: MLXArray) -> MLXArray {
        let w = expandedDimensions(weights, axis: -1)
        let y = switchGLU(x, indices)
        return (w * y).sum(axis: -2)
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

    @ModuleInfo(key: "q_proj") var queryProj: Linear
    @ModuleInfo(key: "k_proj") var keyProj: Linear
    @ModuleInfo(key: "v_proj") var valueProj: Linear?
    @ModuleInfo(key: "o_proj") var outputProj: Linear
    @ModuleInfo(key: "q_norm") var queryNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var keyNorm: RMSNorm
    @ModuleInfo(key: "v_norm") var valueNorm: Gemma4RMSNormNoScale
    @ModuleInfo var rope: RoPELayer

    init(_ config: Gemma4TextConfiguration, layerIdx: Int) {
        self.config = config
        self.layerIdx = layerIdx
        let types = config.resolvedLayerTypes()
        self.layerType = types[layerIdx]
        self.isSliding = layerType == "sliding_attention"
        self.effectiveHeadDim = (!isSliding && (config.globalHeadDim ?? 0) > 0)
            ? config.globalHeadDim! : config.headDim
        let dim = config.hiddenSize
        self.nHeads = config.attentionHeads
        self.useKEqV = config.attentionKEqV && !isSliding
        self.nKVHeads = (useKEqV && config.globalKVHeads != nil)
            ? config.globalKVHeads! : config.kvHeads
        self.scale = 1.0

        self._queryProj.wrappedValue = Linear(dim, nHeads * effectiveHeadDim, bias: false)
        self._keyProj.wrappedValue = Linear(dim, nKVHeads * effectiveHeadDim, bias: false)
        if !useKEqV {
            self._valueProj.wrappedValue = Linear(dim, nKVHeads * effectiveHeadDim, bias: false)
        }
        self._outputProj.wrappedValue = Linear(nHeads * effectiveHeadDim, dim, bias: false)
        self._queryNorm.wrappedValue = RMSNorm(dimensions: effectiveHeadDim, eps: config.rmsNormEps)
        self._keyNorm.wrappedValue = RMSNorm(dimensions: effectiveHeadDim, eps: config.rmsNormEps)
        self._valueNorm.wrappedValue = Gemma4RMSNormNoScale(dimensions: effectiveHeadDim, eps: config.rmsNormEps)

        let layerKey = isSliding ? "sliding_attention" : "full_attention"
        let ropeParams = config.ropeParameters?[layerKey]
        let ropeTheta = ropeParams?["rope_theta"]?.asFloat() ?? (isSliding ? 10_000.0 : 1_000_000.0)
        self.rope = initializeRope(
            dims: effectiveHeadDim, base: ropeTheta,
            traditional: config.ropeTraditional,
            scalingConfig: ropeParams, maxPositionEmbeddings: config.maxPositionEmbeddings
        )
        super.init()
    }

    /// Returns (output, (keys, values), offset) — matching Python exactly
    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache? = nil,
        sharedKV: (MLXArray, MLXArray)? = nil,
        offset: Int = 0
    ) -> (MLXArray, (MLXArray, MLXArray), Int) {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

        var queries = queryProj(x).reshaped(B, L, nHeads, effectiveHeadDim)
        queries = queryNorm(queries)

        var keys: MLXArray
        var values: MLXArray
        var effectiveOffset = offset

        if let (sk, sv) = sharedKV {
            keys = sk
            values = sv
        } else {
            keys = keyProj(x).reshaped(B, L, nKVHeads, effectiveHeadDim)
            values = useKEqV ? keys : (valueProj?(x).reshaped(B, L, nKVHeads, effectiveHeadDim) ?? keys)

            effectiveOffset = cache?.offset ?? 0

            keys = keyNorm(keys)
            keys = keys.transposed(0, 2, 1, 3)
            keys = rope(keys, offset: effectiveOffset)

            values = valueNorm(values)
            values = values.transposed(0, 2, 1, 3)
        }

        queries = queries.transposed(0, 2, 1, 3)
        queries = rope(queries, offset: effectiveOffset)

        if let cache {
            let (ck, cv) = cache.update(keys: keys, values: values)
            keys = ck
            values = cv
        }

        let output = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return (outputProj(output), (keys, values), effectiveOffset)
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
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm: RMSNorm

    @ModuleInfo var router: Gemma4Router?
    @ModuleInfo var experts: Gemma4Experts?
    @ModuleInfo(key: "pre_feedforward_layernorm_2") var preFeedforwardLayerNorm2: RMSNorm?
    @ModuleInfo(key: "post_feedforward_layernorm_1") var postFeedforwardLayerNorm1: RMSNorm?
    @ModuleInfo(key: "post_feedforward_layernorm_2") var postFeedforwardLayerNorm2: RMSNorm?

    @ModuleInfo(key: "per_layer_input_gate") var perLayerInputGate: Linear?
    @ModuleInfo(key: "per_layer_projection") var perLayerProjection: Linear?
    @ModuleInfo(key: "post_per_layer_input_norm") var postPerLayerInputNorm: RMSNorm?

    @ParameterInfo(key: "layer_scalar") var layerScalar: MLXArray

    init(_ config: Gemma4TextConfiguration, layerIdx: Int) {
        self.config = config
        self.layerIdx = layerIdx
        let types = config.resolvedLayerTypes()
        self.layerType = types[layerIdx]
        self.enableMoE = config.enableMoeBlock
        self.hasPLE = config.hiddenSizePerLayerInput > 0

        self._selfAttention.wrappedValue = Gemma4Attention(config, layerIdx: layerIdx)

        let isKVShared = layerIdx >= config.firstKVSharedLayerIdx && config.firstKVSharedLayerIdx > 0
        let effIntermediate = (config.useDoubleWideMLP && isKVShared)
            ? config.intermediateSize * 2 : config.intermediateSize
        self.mlp = Gemma4MLP(hiddenSize: config.hiddenSize, intermediateSize: effIntermediate)

        self._inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._preFeedforwardLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postFeedforwardLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        if enableMoE {
            self._router.wrappedValue = Gemma4Router(config)
            self._experts.wrappedValue = Gemma4Experts(config)
            self._preFeedforwardLayerNorm2.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
            self._postFeedforwardLayerNorm1.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
            self._postFeedforwardLayerNorm2.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }

        if hasPLE {
            let pleSize = config.hiddenSizePerLayerInput
            self._perLayerInputGate.wrappedValue = Linear(config.hiddenSize, pleSize, bias: false)
            self._perLayerProjection.wrappedValue = Linear(pleSize, config.hiddenSize, bias: false)
            self._postPerLayerInputNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }

        self._layerScalar.wrappedValue = MLXArray.ones([1])
        super.init()
    }

    /// Returns (h, shared_kv, offset) — matching Python exactly
    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache? = nil,
        perLayerInput: MLXArray? = nil,
        sharedKV: (MLXArray, MLXArray)? = nil,
        offset: Int = 0
    ) -> (MLXArray, (MLXArray, MLXArray)?, Int) {
        var residual = x
        var h = inputLayerNorm(x)
        let (attnOut, kvs, newOffset) = selfAttention(h, mask: mask, cache: cache, sharedKV: sharedKV, offset: offset)
        h = postAttentionLayerNorm(attnOut)
        h = residual + h

        residual = h
        if enableMoE, let router, let experts,
           let preNorm2 = preFeedforwardLayerNorm2,
           let postNorm1 = postFeedforwardLayerNorm1,
           let postNorm2 = postFeedforwardLayerNorm2 {
            var h1 = preFeedforwardLayerNorm(h)
            h1 = mlp(h1)
            h1 = postNorm1(h1)
            let (topKIdx, topKW) = router(h)
            var h2 = preNorm2(h)
            h2 = experts(h2, indices: topKIdx, weights: topKW)
            h2 = postNorm2(h2)
            h = h1 + h2
        } else {
            h = preFeedforwardLayerNorm(h)
            h = mlp(h)
        }
        h = postFeedforwardLayerNorm(h)
        h = residual + h

        if hasPLE, let gate = perLayerInputGate, let proj = perLayerProjection,
           let postNorm = postPerLayerInputNorm, let pli = perLayerInput {
            residual = h
            var g = gate(h)
            g = geluApproximate(g)
            g = g * pli
            g = proj(g)
            g = postNorm(g)
            h = residual + g
        }

        h = h * layerScalar
        return (h, kvs, newOffset)
    }
}

// MARK: - Text Model

public class Gemma4Model: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [Gemma4DecoderLayer]
    @ModuleInfo var norm: RMSNorm

    // PLE
    @ModuleInfo(key: "embed_tokens_per_layer") var embedTokensPerLayer: Embedding?
    @ModuleInfo(key: "per_layer_model_projection") var perLayerModelProjection: Linear?
    @ModuleInfo(key: "per_layer_projection_norm") var perLayerProjectionNorm: RMSNorm?

    let config: Gemma4TextConfiguration
    let embedScale: Float
    let hasPLE: Bool
    let pleScale: Float
    let pleEmbedScale: Float
    let perLayerProjectionScale: Float
    let previousKVs: [Int]

    init(_ config: Gemma4TextConfiguration) {
        self.config = config
        self.embedScale = sqrt(Float(config.hiddenSize))
        self.hasPLE = config.hiddenSizePerLayerInput > 0
        self.pleScale = pow(2.0, -0.5)
        self.pleEmbedScale = sqrt(Float(max(config.hiddenSizePerLayerInput, 1)))
        self.perLayerProjectionScale = pow(Float(config.hiddenSize), -0.5)

        self._embedTokens.wrappedValue = Embedding(embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        self._layers.wrappedValue = (0 ..< config.hiddenLayers).map { Gemma4DecoderLayer(config, layerIdx: $0) }
        self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        if hasPLE {
            let pleVocab = config.vocabSizePerLayerInput ?? config.vocabularySize
            let pleTotalDim = config.hiddenLayers * config.hiddenSizePerLayerInput
            self._embedTokensPerLayer.wrappedValue = Embedding(embeddingCount: pleVocab, dimensions: pleTotalDim)
            self._perLayerModelProjection.wrappedValue = Linear(config.hiddenSize, pleTotalDim, bias: false)
            self._perLayerProjectionNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSizePerLayerInput, eps: config.rmsNormEps)
        }

        // Build shared KV mapping (Python: self.previous_kvs)
        let types = config.resolvedLayerTypes()
        let M = config.firstKVSharedLayerIdx
        var mapping = Array(0 ..< config.hiddenLayers)
        if config.numKVSharedLayers > 0 {
            var kvsByType: [String: Int] = [:]
            for i in 0 ..< M { kvsByType[types[i]] = i }
            for j in M ..< config.hiddenLayers {
                mapping[j] = kvsByType[types[j]] ?? j
            }
        }
        self.previousKVs = mapping

        super.init()
    }

    private func getPerLayerInputs(_ inputIds: MLXArray) -> MLXArray {
        guard let embed = embedTokensPerLayer else { fatalError("PLE not init") }
        var result = embed(inputIds)
        result = result * pleEmbedScale
        return result.reshaped(inputIds.dim(0), inputIds.dim(1), config.hiddenLayers, config.hiddenSizePerLayerInput)
    }

    private func projectPerLayerInputs(_ h: MLXArray, _ perLayerInputs: MLXArray?) -> MLXArray {
        guard let proj = perLayerModelProjection, let normLayer = perLayerProjectionNorm else { fatalError("PLE not init") }
        var plp = proj(h)
        plp = plp * perLayerProjectionScale
        plp = plp.reshaped(h.dim(0), h.dim(1), config.hiddenLayers, config.hiddenSizePerLayerInput)
        plp = normLayer(plp)
        guard let pli = perLayerInputs else { return plp }
        return (plp + pli) * pleScale
    }

    func callAsFunction(
        _ inputs: MLXArray,
        cache: [KVCache?]? = nil,
        inputEmbeddings: MLXArray? = nil,
        perLayerInputs: MLXArray? = nil
    ) -> MLXArray {
        var h: MLXArray
        if let ie = inputEmbeddings { h = ie } else { h = embedTokens(inputs) }
        h = h * embedScale

        // PLE
        var pliList: [MLXArray?] = Array(repeating: nil, count: config.hiddenLayers)
        if hasPLE {
            let pli = perLayerInputs ?? getPerLayerInputs(inputs)
            let projected = projectPerLayerInputs(h, pli)
            for i in 0 ..< config.hiddenLayers {
                pliList[i] = projected[0..., 0..., i, 0...]
            }
        }

        // Cache: pad with nil for shared layers
        var layerCache: [KVCache?]
        if let c = cache {
            layerCache = c + Array(repeating: nil as KVCache?, count: config.hiddenLayers - c.count)
        } else {
            layerCache = Array(repeating: nil, count: config.hiddenLayers)
        }

        // Masks
        let types = config.resolvedLayerTypes()
        var maskCache: [String: MLXFast.ScaledDotProductAttentionMaskMode] = [:]
        var masks: [MLXFast.ScaledDotProductAttentionMaskMode] = []
        for (i, layer) in layers.enumerated() {
            let lt = types[i]
            if maskCache[lt] == nil {
                if lt == "full_attention" {
                    maskCache[lt] = createAttentionMask(h: h, cache: layerCache[i])
                } else {
                    maskCache[lt] = createAttentionMask(h: h, cache: layerCache[i], windowSize: config.slidingWindow)
                }
            }
            masks.append(maskCache[lt]!)
        }

        // Forward through layers with shared KV intermediates
        var intermediates: [(kv: (MLXArray, MLXArray)?, offset: Int)] = Array(repeating: (nil, 0), count: config.hiddenLayers)

        for (idx, layer) in layers.enumerated() {
            let prevIdx = previousKVs[idx]
            let sharedKV = intermediates[prevIdx].kv
            let prevOffset = intermediates[prevIdx].offset

            let (newH, kvs, newOffset) = layer(
                h, mask: masks[idx], cache: layerCache[idx],
                perLayerInput: pliList[idx],
                sharedKV: sharedKV, offset: prevOffset
            )
            h = newH
            intermediates[idx] = (kvs, newOffset)
        }

        return norm(h)
    }
}

// MARK: - Top-level Model

public class Gemma4TextModel: Module, LLMModel {
    @ModuleInfo public var model: Gemma4Model
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public let config: Gemma4TextConfiguration
    public var vocabularySize: Int { config.vocabularySize }

    public init(_ config: Gemma4TextConfiguration) {
        self.config = config
        self.model = Gemma4Model(config)
        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        }
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var out = model(inputs, cache: cache)
        if config.tieWordEmbeddings {
            out = model.embedTokens.asLinear(out)
        } else if let head = lmHead {
            out = head(out)
        }
        if let cap = config.finalLogitSoftcapping {
            out = tanh(out / cap) * cap
        }
        return out
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var w = weights

        // Strip language_model prefix (VLM models)
        let unflattened = ModuleParameters.unflattened(weights)
        if let lm = unflattened["language_model"] {
            w = Dictionary(uniqueKeysWithValues: lm.flattened())
        }

        var sanitized: [String: MLXArray] = [:]
        for (k, v) in w {
            if k.contains("self_attn.rotary_emb") { continue }
            if k.contains("input_max") || k.contains("input_min")
                || k.contains("output_max") || k.contains("output_min") { continue }

            // When tie_word_embeddings=true, skip lm_head weights entirely
            // (model uses embed_tokens.as_linear instead)
            if config.tieWordEmbeddings && k.hasPrefix("lm_head.") { continue }

            // MoE expert weight remapping
            if k.hasSuffix(".experts.gate_up_proj") {
                let base = String(k.dropLast(".gate_up_proj".count))
                let parts = split(v, parts: 2, axis: -2)
                sanitized["\(base).switch_glu.gate_proj.weight"] = parts[0]
                sanitized["\(base).switch_glu.up_proj.weight"] = parts[1]
                continue
            }
            if k.hasSuffix(".experts.down_proj") {
                let base = String(k.dropLast(".down_proj".count))
                sanitized["\(base).switch_glu.down_proj.weight"] = v
                continue
            }

            sanitized[k] = v
        }

        return sanitized
    }

    public func newCache(parameters: GenerateParameters? = nil) -> [KVCache] {
        let types = config.resolvedLayerTypes()
        let numConcrete = config.firstKVSharedLayerIdx
        var caches = [KVCache]()
        for i in 0 ..< numConcrete {
            if types[i] == "full_attention" {
                caches.append(StandardKVCache())
            } else {
                caches.append(RotatingKVCache(maxSize: config.slidingWindow, keep: 0))
            }
        }
        return caches
    }

    public func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int? = nil) throws -> PrepareResult {
        guard input.text.tokens.shape[0] > 0 else {
            return .tokens(.init(tokens: MLXArray(Int32(0))[0 ..< 0]))
        }
        return .tokens(input.text)
    }
}

extension Gemma4TextModel: LoRAModel {
    public var loraLayers: [Module] { model.layers }
}
