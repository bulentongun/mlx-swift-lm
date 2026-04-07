//
//  RoPEUtils.swift
//  mlx-swift-lm
//
//  Created by John Mai on 2025/8/11.
//

import Foundation
import MLX
import MLXNN

public class Llama3RoPE: Module, OffsetLayer, ArrayOffsetLayer {
    let dims: Int
    let maxPositionEmbeddings: Int
    let traditional: Bool
    let _freqs: MLXArray

    init(
        dims: Int,
        maxPositionEmbeddings: Int = 2048,
        traditional: Bool = false,
        base: Float = 10000,
        scalingConfig: [String: StringOrNumber]? = nil
    ) {
        self.dims = dims
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.traditional = traditional

        guard let scalingConfig = scalingConfig else {
            fatalError("Llama3RoPE requires scaling_config")
        }

        let factor = scalingConfig["factor"]?.asFloat() ?? 1.0
        let lowFreqFactor = scalingConfig["low_freq_factor"]?.asFloat() ?? 1.0
        let highFreqFactor = scalingConfig["high_freq_factor"]?.asFloat() ?? 4.0
        let oldContextLen = scalingConfig["original_max_position_embeddings"]?.asFloat() ?? 8192.0

        let lowFreqWavelen = oldContextLen / lowFreqFactor
        let highFreqWavelen = oldContextLen / highFreqFactor

        let indices = MLXArray(stride(from: 0, to: dims, by: 2))
        var frequencies = MLX.pow(base, indices / Float(dims))
        let wavelens = 2 * Float.pi * frequencies

        frequencies = MLX.where(
            wavelens .> MLXArray(lowFreqWavelen),
            frequencies * factor,
            frequencies
        )

        let isMediumFreq = MLX.logicalAnd(
            wavelens .> MLXArray(highFreqWavelen),
            wavelens .< MLXArray(lowFreqWavelen)
        )

        let smoothFactors =
            (oldContextLen / wavelens - lowFreqFactor) / (highFreqFactor - lowFreqFactor)
        let smoothFreqs = frequencies / ((1 - smoothFactors) / factor + smoothFactors)

        self._freqs = MLX.where(isMediumFreq, smoothFreqs, frequencies)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        MLXFast.RoPE(
            x,
            dimensions: dims,
            traditional: traditional,
            base: nil,
            scale: 1.0,
            offset: offset,
            freqs: _freqs
        )
    }

    public func callAsFunction(_ x: MLXArray, offset: MLXArray) -> MLXArray {
        MLXFast.RoPE(
            x,
            dimensions: dims,
            traditional: traditional,
            base: nil,
            scale: 1.0,
            offset: offset,
            freqs: _freqs
        )
    }

}

public class YarnRoPE: Module, OffsetLayer, ArrayOffsetLayer {
    let dimensions: Int
    let traditional: Bool

    private let _mscale: Float
    private let _freqs: MLXArray

    public init(
        dimensions: Int,
        traditional: Bool = false,
        maxPositionEmbeddings: Int = 2048,
        base: Float = 10000,
        scalingFactor: Float = 1.0,
        originalMaxPositionEmbeddings: Int = 4096,
        betaFast: Float = 32,
        betaSlow: Float = 1,
        mscale: Float = 1,
        mscaleAllDim: Float = 0
    ) {
        precondition(dimensions % 2 == 0, "Dimensions must be even")

        self.dimensions = dimensions
        self.traditional = traditional

        func yarnFindCorrectionDim(numRotations: Float) -> Float {
            return Float(dimensions)
                * log(Float(originalMaxPositionEmbeddings) / (numRotations * 2 * Float.pi))
                / (2 * log(base))
        }

        func yarnFindCorrectionRange() -> (low: Int, high: Int) {
            let low = Int(floor(yarnFindCorrectionDim(numRotations: betaFast)))
            let high = Int(ceil(yarnFindCorrectionDim(numRotations: betaSlow)))
            return (max(low, 0), min(high, dimensions - 1))
        }

        func yarnGetMscale(scale: Float, mscale: Float) -> Float {
            if scale <= 1 {
                return 1.0
            }
            return 0.1 * mscale * log(scale) + 1.0
        }

        func yarnLinearRampMask(minVal: Float, maxVal: Float, dim: Int) -> MLXArray {
            var maxVal = maxVal
            if minVal == maxVal {
                maxVal += 0.001
            }

            let linearFunc = (MLXArray(0 ..< dim).asType(.float32) - minVal) / (maxVal - minVal)
            return clip(linearFunc, min: 0, max: 1)
        }

        self._mscale =
            yarnGetMscale(scale: scalingFactor, mscale: mscale)
            / yarnGetMscale(scale: scalingFactor, mscale: mscaleAllDim)

        let freqExtra = pow(
            base,
            MLXArray(stride(from: 0, to: dimensions, by: 2)).asType(.float32)
                / dimensions)
        let freqInter =
            scalingFactor
            * pow(
                base,
                MLXArray(stride(from: 0, to: dimensions, by: 2)).asType(.float32)
                    / dimensions)

        let (low, high) = yarnFindCorrectionRange()
        let freqMask =
            1.0 - yarnLinearRampMask(minVal: Float(low), maxVal: Float(high), dim: dimensions / 2)

        self._freqs = (freqInter * freqExtra) / (freqInter * freqMask + freqExtra * (1 - freqMask))
    }

    public func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        // "copy" of x as we are going to write through it and don't want to update
        // through the reference
        // https://github.com/ml-explore/mlx-swift/issues/364
        var x = x
        if _mscale != 1.0 {
            x = x[0..., .ellipsis]
            x[.ellipsis, 0 ..< dimensions] *= _mscale
        }

        return MLXFast.RoPE(
            x,
            dimensions: dimensions,
            traditional: traditional,
            base: nil,
            scale: 1.0,
            offset: offset,
            freqs: self._freqs
        )
    }

    public func callAsFunction(_ x: MLXArray, offset: MLXArray) -> MLXArray {
        var x = x
        if _mscale != 1.0 {
            x = x[0..., .ellipsis]
            x[.ellipsis, 0 ..< dimensions] *= _mscale
        }

        return MLXFast.RoPE(
            x,
            dimensions: dimensions,
            traditional: traditional,
            base: nil,
            scale: 1.0,
            offset: offset,
            freqs: self._freqs
        )
    }

}

// MARK: - ProportionalRoPE (Gemma 4)

/// Gemma 4 ProportionalRoPE — rotates only a fraction of head dimensions.
/// Frequencies are computed relative to the FULL head dimension (not just the rotated portion).
/// Ported from Blaizzy/mlx-vlm/models/gemma4/rope_utils.py
public class ProportionalRoPE: Module, OffsetLayer, ArrayOffsetLayer {
    let dims: Int
    let rotatedDims: Int
    let traditional: Bool
    let base: Float

    private var _freqs: MLXArray?

    public init(dims: Int, traditional: Bool = false, base: Float = 10000.0, partialRotaryFactor: Float = 1.0) {
        self.dims = dims
        self.traditional = traditional
        self.base = base

        // Calculate how many dimensions to actually rotate
        let ropeAngles = Int(partialRotaryFactor * Float(dims) / 2.0)
        self.rotatedDims = 2 * ropeAngles

        super.init()
    }

    private func computeFreqs() -> MLXArray {
        // Exponents scaled by FULL dims (not rotatedDims) — this is the "proportional" part
        let exponents = MLXArray(stride(from: Float(0), to: Float(rotatedDims), by: 2)) / Float(dims)
        return pow(MLXArray(base), -exponents)
    }

    public func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        let freqs = _freqs ?? computeFreqs()
        if _freqs == nil { _freqs = freqs }
        return applyProportionalRoPE(x, freqs: freqs, offset: offset)
    }

    public func callAsFunction(_ x: MLXArray, offset: MLXArray) -> MLXArray {
        let freqs = _freqs ?? computeFreqs()
        if _freqs == nil { _freqs = freqs }
        return applyProportionalRoPE(x, freqs: freqs, offsetArray: offset)
    }

    private func applyProportionalRoPE(_ x: MLXArray, freqs: MLXArray, offset: Int = 0, offsetArray: MLXArray? = nil) -> MLXArray {
        // x shape: [B, nHeads, L, headDim]
        let shape = x.shape
        let headDim = shape[shape.count - 1]

        if rotatedDims >= headDim {
            // Full rotation — use standard RoPE
            return MLXFast.RoPE(
                x, dimensions: rotatedDims, traditional: traditional,
                base: nil, scale: 1.0, offset: offset, freqs: freqs
            )
        }

        // Partial rotation: split into rotated and passthrough parts
        // Split at rotatedDims/2 from each half (HF rotate_half convention)
        let halfDim = headDim / 2
        let ropeAngles = rotatedDims / 2

        // Split x into left half and right half
        let xLeft = x[.ellipsis, ..<halfDim]
        let xRight = x[.ellipsis, halfDim...]

        // Extract the portions to rotate from each half
        let xLeftRot = xLeft[.ellipsis, ..<ropeAngles]
        let xLeftPass = xLeft[.ellipsis, ropeAngles...]
        let xRightRot = xRight[.ellipsis, ..<ropeAngles]
        let xRightPass = xRight[.ellipsis, ropeAngles...]

        // Apply RoPE to rotated portions (concatenated)
        let toRotate = concatenated([xLeftRot, xRightRot], axis: -1)
        let rotated = MLXFast.RoPE(
            toRotate, dimensions: rotatedDims, traditional: traditional,
            base: nil, scale: 1.0, offset: offset, freqs: freqs
        )

        // Split rotated back
        let rotatedLeft = rotated[.ellipsis, ..<ropeAngles]
        let rotatedRight = rotated[.ellipsis, ropeAngles...]

        // Reconstruct
        let newLeft = concatenated([rotatedLeft, xLeftPass], axis: -1)
        let newRight = concatenated([rotatedRight, xRightPass], axis: -1)

        return concatenated([newLeft, newRight], axis: -1)
    }
}

private let yarnTypes: Set = ["yarn", "deepseek_yarn", "telechat3-yarn"]

public typealias RoPELayer = OffsetLayer & ArrayOffsetLayer

public func initializeRope(
    dims: Int,
    base: Float,
    traditional: Bool,
    scalingConfig: [String: StringOrNumber]?,
    maxPositionEmbeddings: Int?
) -> RoPELayer {
    let ropeType: String = {
        if let config = scalingConfig,
            let typeValue = config["type"] ?? config["rope_type"],
            case .string(let s) = typeValue
        {
            return s
        }
        return "default"
    }()

    if ropeType == "default" || ropeType == "linear" {
        let scale: Float
        if ropeType == "linear", let factor = scalingConfig?["factor"]?.asFloat() {
            scale = 1 / factor
        } else {
            scale = 1.0
        }
        return RoPE(dimensions: dims, traditional: traditional, base: base, scale: scale)
    } else if ropeType == "llama3" {
        return Llama3RoPE(
            dims: dims,
            maxPositionEmbeddings: maxPositionEmbeddings ?? 2048,
            traditional: traditional,
            base: base,
            scalingConfig: scalingConfig
        )
    } else if yarnTypes.contains(ropeType) {
        let factor = scalingConfig?["factor"]?.asFloat() ?? 32.0
        let origMax = scalingConfig?["original_max_position_embeddings"]?.asInt() ?? 4096
        let betaFast = scalingConfig?["beta_fast"]?.asFloat() ?? 32.0
        let betaSlow = scalingConfig?["beta_slow"]?.asFloat() ?? 1.0
        let mscale = scalingConfig?["mscale"]?.asFloat() ?? 1.0
        let mscaleAllDim = scalingConfig?["mscale_all_dim"]?.asFloat() ?? 0.0

        return YarnRoPE(
            dimensions: dims,
            traditional: traditional,
            maxPositionEmbeddings: maxPositionEmbeddings ?? 2048,
            base: base,
            scalingFactor: factor,
            originalMaxPositionEmbeddings: origMax,
            betaFast: betaFast,
            betaSlow: betaSlow,
            mscale: mscale,
            mscaleAllDim: mscaleAllDim
        )
    } else if ropeType == "longrope" {
        guard let config = scalingConfig else {
            fatalError("longrope requires scaling_config")
        }
        guard let origMax = config["original_max_position_embeddings"]?.asInt() else {
            fatalError("longrope requires original_max_position_embeddings")
        }
        guard let shortFactor = config["short_factor"]?.asFloats() else {
            fatalError("longrope requires short_factor")
        }
        guard let longFactor = config["long_factor"]?.asFloats() else {
            fatalError("longrope requires long_factor")
        }

        return SuScaledRoPE(
            dimensions: dims,
            base: base,
            maxPositionEmbeddings: maxPositionEmbeddings ?? 131072,
            originalMaxPositionEmbeddings: origMax,
            shortFactor: shortFactor,
            longFactor: longFactor
        )
    } else if ropeType == "proportional" {
        // Gemma 4 ProportionalRoPE — partial rotation with frequencies scaled by full dim
        let partialFactor = scalingConfig?["partial_rotary_factor"]?.asFloat() ?? 1.0
        return ProportionalRoPE(
            dims: dims,
            traditional: traditional,
            base: base,
            partialRotaryFactor: partialFactor
        )
    } else if ropeType == "mrope" {
        // MRoPE returns basic RoPE here. The actual multi-modal rotary embedding logic
        // (applying different embeddings per modality) is handled in the attention layer
        // of multimodal models like Qwen2VL, not in the RoPE module itself.
        if let config = scalingConfig,
            let mropeSection = config["mrope_section"]?.asInts()
        {
            precondition(
                mropeSection.count == 3,
                "MRoPE currently only supports 3 sections, got \(mropeSection.count)"
            )
        }
        return RoPE(dimensions: dims, traditional: traditional, base: base, scale: 1.0)
    } else {
        fatalError("Unsupported RoPE type: \(ropeType)")
    }
}
