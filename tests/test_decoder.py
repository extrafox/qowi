import numpy as np
import unittest
from bitstring import BitStream
from qowi.decoder import Decoder
from qowi.encoder import Encoder
from qowi.wavelet import Wavelet

TEST_IMAGES = [
    np.array([
        [[255, 255, 255], [0, 0, 0]],
        [[0, 0, 0], [255, 255, 255]],
    ], dtype='uint8'),
    np.array([
        [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[4, 4, 4], [4, 4, 4], [4, 4, 4], [4, 4, 4]],
        [[8, 8, 8], [8, 8, 8], [8, 8, 8], [8, 8, 8]],
        [[16, 16, 16], [16, 16, 16], [16, 16, 16], [16, 16, 16]],
    ], dtype='uint8'),
    np.array([
        [[95, 108, 109], [97, 109, 108], [93, 106, 109], [95, 111, 115]],
        [[91, 85, 76], [83, 75, 65], [91, 93, 88], [83, 85, 80]],
        [[83, 73, 63], [74, 66, 56], [80, 87, 86], [79, 87, 87]],
        [[195, 181, 167], [173, 164, 152], [75, 82, 82], [77, 88, 90]],
    ], dtype='uint8'),
    np.array([
        [[95, 108, 109], [97, 109, 108], [93, 106, 109], [95, 111, 115], [180, 202, 215], [170, 195, 214], [175, 206, 225], [189, 216, 233], [197, 213, 225], [173, 197, 204], [113, 124, 124], [147, 165, 169], [156, 176, 182], [150, 171, 175], [118, 134, 135], [115, 128, 127]],
        [[91, 85, 76], [83, 75, 65], [91, 93, 88], [83, 85, 80], [179, 199, 211], [176, 200, 220], [177, 206, 225], [191, 216, 233], [197, 214, 227], [171, 194, 202], [118, 134, 138], [134, 159, 168], [137, 159, 165], [139, 154, 158], [132, 140, 138], [134, 143, 139]],
        [[83, 73, 63], [74, 66, 56], [80, 87, 86], [79, 87, 87], [164, 180, 191], [179, 200, 217], [175, 197, 209], [185, 205, 218], [193, 207, 219], [167, 191, 201], [115, 127, 130], [148, 162, 167], [154, 169, 173], [150, 164, 168], [130, 141, 141], [125, 140, 141]],
        [[195, 181, 167], [173, 164, 152], [75, 82, 82], [77, 88, 90], [162, 177, 187], [177, 197, 212], [170, 186, 194], [179, 194, 203], [185, 196, 207], [157, 182, 193], [107, 125, 132], [130, 153, 162], [131, 154, 163], [126, 146, 153], [108, 123, 125], [113, 127, 128]],
        [[228, 213, 197], [234, 221, 206], [190, 195, 191], [162, 178, 183], [165, 172, 176], [179, 188, 192], [131, 124, 119], [111, 97, 88], [123, 119, 119], [152, 172, 181], [118, 130, 133], [140, 155, 160], [145, 160, 165], [141, 153, 156], [126, 135, 133], [127, 137, 136]],
        [[216, 201, 186], [229, 213, 199], [235, 222, 210], [141, 134, 127], [65, 42, 31], [210, 187, 159], [212, 189, 161], [79, 59, 47], [44, 20, 11], [125, 134, 136], [100, 117, 124], [116, 138, 147], [119, 141, 150], [110, 130, 136], [94, 110, 113], [98, 116, 119]],
        [[190, 171, 154], [219, 206, 194], [67, 65, 61], [84, 73, 57], [101, 84, 66], [187, 160, 134], [177, 148, 124], [133, 117, 104], [130, 114, 99], [197, 196, 190], [203, 202, 194], [190, 195, 191], [181, 186, 184], [177, 181, 177], [159, 161, 155], [150, 155, 151]],
        [[218, 205, 191], [216, 205, 192], [162, 153, 143], [188, 178, 166], [198, 183, 165], [219, 194, 168], [207, 183, 159], [175, 162, 149], [225, 213, 198], [249, 237, 222], [250, 238, 224], [250, 238, 224], [249, 237, 225], [246, 235, 223], [229, 219, 208], [224, 215, 205]],
        [[247, 231, 213], [248, 232, 213], [249, 234, 216], [240, 224, 204], [190, 168, 143], [120, 109, 91], [71, 62, 51], [165, 147, 133], [204, 190, 173], [249, 236, 222], [248, 236, 221], [246, 234, 220], [245, 234, 221], [242, 233, 223], [246, 235, 223], [245, 233, 222]],
        [[252, 236, 217], [250, 232, 213], [244, 225, 203], [213, 187, 161], [156, 132, 106], [196, 176, 149], [132, 110, 89], [138, 112, 88], [188, 171, 152], [245, 231, 214], [245, 232, 217], [245, 232, 217], [246, 233, 218], [246, 232, 219], [244, 231, 218], [240, 227, 216]],
        [[252, 235, 214], [252, 236, 215], [252, 237, 217], [241, 225, 204], [210, 189, 162], [215, 193, 164], [155, 135, 114], [137, 114, 90], [182, 165, 142], [190, 178, 161], [229, 216, 197], [233, 218, 201], [242, 227, 211], [243, 228, 214], [239, 226, 212], [239, 227, 214]],
        [[253, 238, 217], [254, 239, 215], [252, 238, 212], [244, 227, 202], [218, 199, 174], [216, 191, 158], [133, 108, 81], [81, 59, 38], [172, 145, 117], [206, 186, 162], [235, 220, 201], [249, 234, 215], [252, 237, 219], [251, 237, 220], [248, 234, 218], [241, 228, 215]],
        [[251, 237, 218], [244, 230, 211], [237, 221, 201], [237, 219, 195], [211, 192, 167], [232, 209, 179], [233, 209, 179], [198, 179, 155], [218, 199, 175], [192, 167, 142], [229, 204, 176], [244, 226, 205], [250, 234, 214], [251, 234, 215], [248, 232, 215], [245, 231, 217]],
        [[241, 226, 210], [242, 228, 211], [246, 230, 214], [245, 230, 212], [233, 219, 203], [237, 219, 197], [229, 206, 179], [220, 192, 166], [224, 201, 177], [252, 238, 211], [216, 190, 165], [198, 167, 143], [204, 178, 157], [251, 235, 214], [249, 233, 216], [241, 228, 214]],
        [[251, 235, 217], [248, 232, 216], [247, 231, 212], [234, 220, 204], [231, 217, 203], [243, 226, 207], [235, 214, 191], [194, 160, 134], [180, 148, 124], [252, 236, 214], [254, 240, 220], [254, 240, 220], [253, 239, 219], [252, 236, 216], [248, 233, 216], [241, 225, 212]], [[250, 233, 214], [243, 224, 205], [243, 226, 207], [245, 229, 211], [242, 226, 211], [242, 227, 212], [249, 232, 216], [245, 229, 212], [248, 232, 215], [250, 234, 216], [254, 239, 219], [252, 236, 216], [253, 236, 217], [251, 235, 215], [245, 229, 213], [240, 224, 211]]
    ], dtype = 'uint8'),
    np.array([
        [[95, 108, 109], [97, 109, 108], [93, 106, 109], [95, 111, 115]],
        [[91, 85, 76], [83, 75, 65], [91, 93, 88], [83, 85, 80]],
        [[83, 73, 63], [74, 66, 56], [80, 87, 86], [79, 87, 87]],
        [[195, 181, 167], [173, 164, 152], [75, 82, 82], [77, 88, 90]],
    ], dtype = 'uint8'),
    np.array([
        [[235, 222, 210], [141, 134, 127], [65, 42, 31], [210, 187, 159]],
        [[67, 65, 61], [84, 73, 57], [101, 84, 66], [187, 160, 134]],
        [[162, 153, 143], [188, 178, 166], [198, 183, 165], [219, 194, 168]],
        [[249, 234, 216], [240, 224, 204], [190, 168, 143], [120, 109, 91]],
    ], dtype = 'uint8'),
        np.array([
        [[95, 108, 109], [97, 109, 108], [93, 106, 109], [95, 111, 115], [180, 202, 215], [170, 195, 214], [175, 206, 225], [189, 216, 233]],
        [[91, 85, 76], [83, 75, 65], [91, 93, 88], [83, 85, 80], [179, 199, 211], [176, 200, 220], [177, 206, 225], [191, 216, 233]],
        [[83, 73, 63], [74, 66, 56], [80, 87, 86], [79, 87, 87], [164, 180, 191], [179, 200, 217], [175, 197, 209], [185, 205, 218]],
        [[195, 181, 167], [173, 164, 152], [75, 82, 82], [77, 88, 90], [162, 177, 187], [177, 197, 212], [170, 186, 194], [179, 194, 203]],
        [[228, 213, 197], [234, 221, 206], [190, 195, 191], [162, 178, 183], [165, 172, 176], [179, 188, 192], [131, 124, 119], [111, 97, 88]],
        [[216, 201, 186], [229, 213, 199], [235, 222, 210], [141, 134, 127], [65, 42, 31], [210, 187, 159], [212, 189, 161], [79, 59, 47]],
        [[190, 171, 154], [219, 206, 194], [67, 65, 61], [84, 73, 57], [101, 84, 66], [187, 160, 134], [177, 148, 124], [133, 117, 104]],
        [[218, 205, 191], [216, 205, 192], [162, 153, 143], [188, 178, 166], [198, 183, 165], [219, 194, 168], [207, 183, 159], [175, 162, 149]],
    ], dtype = 'uint8'),
]

class TestEncoder(unittest.TestCase):

    def test_instantiation(self):
        d = Decoder(BitStream())
        self.assertIsInstance(d, Decoder)

    def test_instantiation_from_zero_image(self):
        source_image = TEST_IMAGES[0]
        wavelet = Wavelet().prepare_from_image(source_image)
        bits = Encoder(wavelet).encode()
        decoder = Decoder(BitStream(bits))
        self.assertIsInstance(decoder, Decoder)

    def test_round_trip_from_zero_image(self):
        source_image = TEST_IMAGES[0]
        wavelet = Wavelet().prepare_from_image(source_image)
        bits = Encoder(wavelet).encode()
        decoded_image = Decoder(BitStream(bits)).decode()
        self.assertTrue((source_image == decoded_image).all())

    def test_round_trip_from_two_image(self):
        source_image = TEST_IMAGES[2]
        source_wavelet = Wavelet().prepare_from_image(source_image)
        bits = Encoder(source_wavelet).encode()
        decoder = Decoder(BitStream(bits))
        decoded_image = decoder.decode()
        decoder_wavelet = decoder._wavelet
        self.assertTrue((source_image == decoded_image).all())

    def test_round_trip_from_three_image(self):
        source_image = TEST_IMAGES[3]
        source_wavelet = Wavelet().prepare_from_image(source_image)
        bits = Encoder(source_wavelet).encode()
        decoder = Decoder(BitStream(bits))
        decoded_image = decoder.decode()
        decoder_wavelet = decoder._wavelet
        self.assertTrue((source_image == decoded_image).all())

if __name__ == '__main__':
    unittest.main()
