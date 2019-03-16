#!/usr/bin/env python

"""Generate json representations of audio files.

Simple Python script that computes a json data representation of a single
wavefor by first taking the average of the N-channels of the input and then
using linear interpolation to shrink/expand the original audio data to the
requested number of output samples.

Note that this type of interpolation is NOT suitable for audio resampling in
general, but serves to reduce/expand the amount of audio data for visualization
purposes.
"""

import argparse
import os.path
import scipy.io.wavfile
import numpy
import json
import decimal
import math


# parse input arguments
def parseArgs():
    """Check the extension of an audio file."""
    def check_audio_file_ext(allowed):
        class Action(argparse.Action):
            def __call__(self, parser, namespace, fname, option_string=None):
                ext = os.path.splitext(fname)[1][1:]
                if ext not in allowed:
                    option_string = '({})'.format(option_string) if \
                        option_string else ''
                    parser.error(
                        "file extension is not one of {}{}"
                        .format(allowed, option_string)
                    )
                else:
                    setattr(namespace, self.dest, fname)

        return Action

    """Check if the precision is in the allowed range."""
    def check_precision_range(prec_range):
        class Action(argparse.Action):
            def __call__(self, parser, namespace, prec, option_string=None):
                if prec not in xrange(*prec_range):
                    option_string = '({})'.format(option_string) if \
                        option_string else ''
                    parser.error(
                        "float precision is not in range [{}, {}]{}"
                        .format(
                            prec_range[0], prec_range[1] - 1, option_string
                        )
                    )
                else:
                    setattr(namespace, self.dest, prec)

        return Action

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--ifile",
        action=check_audio_file_ext({'wav'}),
        help="Path to input file",
        required=True
    )
    parser.add_argument(
        "-o", "--ofile",
        action=check_audio_file_ext({'json'}),
        help="Path to output file in JSON format"
    )
    parser.add_argument(
        "-s", "--samples",
        type=int,
        help="Number of sample points for the waveform representation",
        default=800
    )
    parser.add_argument(
        "-p", "--precision",
        action=check_precision_range((1, 7)),
        type=int,
        help="Precision of the floats representing the waveform amplitude \
            [1, 6]",
        default=6
    )
    parser.add_argument(
        "-n", "--normalize",
        action="store_true",
        help="If set, waveform amplitudes will be normalized to unity"
    )
    parser.add_argument(
        "-l", "--logarithmic",
        action="store_true",
        help="If set, use a logarithmic (e.g. decibel) scale for the waveform \
            amplitudes"
    )

    args = parser.parse_args()

    if args.ofile is None:  # use path of input if no output path is specified
        args.ofile = os.path.splitext(args.ifile)[0] + ".json"

    return args


def lin2log(val):
    """Convert linear amplitude values to logarithmic.

    Compute amplitude in decibel and map it to the range -1.0 to 1.0.
    (clip amplitudes to range -60dB - 0dB)
    """
    db = (3.0 + math.log10(min(max(abs(val), 0.001), 1.0))) / 3.0
    if val < 0:
        db *= -1
    return db

if __name__ == "__main__":
    args = parseArgs()
    N = args.samples                                # nr. of samples in output
    SR, data = scipy.io.wavfile.read(args.ifile)
    if(data.ndim==1):                            #if the numpy array is 1D,it converts into a 2D numpy array. 
        data = numpy.reshape(data, (-1, 2))       
    M, numChannels = data.shape                     # nr. of samples in input

    # convert fixed point audio data to floating point range -1. to 1.
    if data.dtype == 'int16':
        data = data / (2. ** 15)
    elif data.dtype == 'int32':
        data = data / (2. ** 31)

    # Get nr. of samples of waveform data from the input (note: this is NOT \
    # the way to do proper audio resampling, but will do for visualization \
    # purposes)
    data = data.T
    if numChannels > 1:
        x = numpy.arange(0, M, float(M) / N)
        xp = numpy.arange(M)
        out = numpy.zeros((numChannels, x.size))
        # First interpolate all individuals channels
        for n in xrange(numChannels):
            out[n, :] = numpy.interp(x, xp, data[n, :])
        # Then compute average of n channels
        out = numpy.sum(out, 0) / numChannels
    else:
        out = numpy.interp(
            numpy.arange(0, M, float(M) / N), numpy.arange(M), data
        )

    if args.logarithmic:
        for i in xrange(len(out)):
            out[i] = lin2log(out[i])

    if args.normalize:
        out /= numpy.max(abs(out))

    # dump the waveform data as JSON file
    with open(args.ofile, 'w') as outfile:
        json.dump(
            {
                'data': [
                    float(
                        decimal.Decimal("%.{}f".format(args.precision) % item)
                    ) for item in list(out)
                ]
            }, outfile
        )

    print "JSON file written to disk"
