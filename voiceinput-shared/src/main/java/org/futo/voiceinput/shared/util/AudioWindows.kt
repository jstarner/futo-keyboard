package org.futo.voiceinput.shared.util

private const val SAMPLE_RATE = 16000

// Clips are cut at the quietest 100ms frame within the last 2 seconds of each window
private const val SPLIT_SEARCH_SAMPLES = 2 * SAMPLE_RATE
private const val SPLIT_FRAME_SAMPLES = SAMPLE_RATE / 10

/** Splits 16 kHz audio into windows of at most maxWindowSamples, cutting at quiet points */
fun splitIntoWindows(samples: FloatArray, maxWindowSamples: Int): List<IntRange> {
    val windows = mutableListOf<IntRange>()

    var start = 0
    while(samples.size - start > maxWindowSamples) {
        val limit = start + maxWindowSamples
        val end = quietestCut(samples, limit - SPLIT_SEARCH_SAMPLES, limit)
        windows.add(start until end)
        start = end
    }

    if(start < samples.size) {
        windows.add(start until samples.size)
    }

    return windows
}

// Returns the middle of the lowest-energy frame in [from, to)
private fun quietestCut(samples: FloatArray, from: Int, to: Int): Int {
    var best = to
    var bestEnergy = Double.MAX_VALUE

    var frameStart = from
    while(frameStart + SPLIT_FRAME_SAMPLES <= to) {
        var energy = 0.0
        for(i in frameStart until frameStart + SPLIT_FRAME_SAMPLES) {
            energy += samples[i] * samples[i]
        }

        if(energy < bestEnergy) {
            bestEnergy = energy
            best = frameStart + SPLIT_FRAME_SAMPLES / 2
        }

        frameStart += SPLIT_FRAME_SAMPLES / 2
    }

    return best
}
