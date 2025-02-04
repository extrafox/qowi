import numpy as np
import unittest
from bitstring import BitStream
from qowi.qowi_decoder import QOWIDecoder
from qowi.qowi_encoder import QOWIEncoder

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
        [[251, 235, 217], [248, 232, 216], [247, 231, 212], [234, 220, 204], [231, 217, 203], [243, 226, 207], [235, 214, 191], [194, 160, 134], [180, 148, 124], [252, 236, 214], [254, 240, 220], [254, 240, 220], [253, 239, 219], [252, 236, 216], [248, 233, 216], [241, 225, 212]],
        [[250, 233, 214], [243, 224, 205], [243, 226, 207], [245, 229, 211], [242, 226, 211], [242, 227, 212], [249, 232, 216], [245, 229, 212], [248, 232, 215], [250, 234, 216], [254, 239, 219], [252, 236, 216], [253, 236, 217], [251, 235, 215], [245, 229, 213], [240, 224, 211]]
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
    np.array([
        [[218, 111,  28],  [126, 238,  24],  [198,  57, 226],  [ 75, 132,  92],  [ 11,  11,  55],  [ 19,  25, 194],  [112,  87,  28],  [247, 236,  12],  [ 55, 211,   0],  [149,  70, 195],  [ 10,  89, 201],  [146,  78, 195],  [200, 181, 156],  [195,  54, 207],  [231, 228, 173],  [ 53, 249,  21]],
        [[245,  10, 212],  [121, 137, 123],  [193,  28, 232],  [146,   8, 105],  [192, 111,  72],  [250, 223, 112],  [ 71, 110,  56],  [ 58,  88, 143],  [205, 124, 152],  [104, 154, 113],  [108, 226,  60],  [ 45,  14, 113],  [ 93,  28, 227],  [170, 237,  84],  [201, 115,  15],  [ 91,  60, 161]],
        [[ 18, 187, 180],  [212, 191,  30],  [ 41,  68, 200],  [ 48, 105,  17],  [ 90, 134, 229],  [ 92, 188, 216],  [174, 230,  40],  [ 63, 152,  23],  [209, 182, 112],  [126,  55,  69],  [252,  67, 201],  [111, 104, 251],  [230,  27,  33],  [128,  14,   1],  [ 72, 103,  89],  [ 76, 222,  18]],
        [[175, 147,  92],  [245, 118, 226],  [238,  27,   4],  [212, 242,  14],  [181, 211,   8],  [165, 126, 137],  [226,  64,  31],  [114, 139,  85],  [254, 156,   7],  [208,  46,  16],  [ 58, 226,  70],  [ 84,  89, 203],  [156,  66, 141],  [ 91, 178,  13],  [213, 202, 242],  [112, 171,   5]],
        [[171,  17, 237],  [ 47, 142, 203],  [108, 202, 124],  [ 33,  69,  18],  [127,  12,  24],  [ 84, 211, 182],  [233, 132, 111],  [ 98, 188, 194],  [182,  85, 199],  [ 29,  74,  94],  [177, 239, 183],  [154, 167,  96],  [152,  88, 126],  [170, 183, 190],  [166, 233, 250],  [220,  50, 235]],
        [[223, 110,  20],  [255,   3, 100],  [181,  33, 250],  [ 69, 194, 140],  [ 99,  57, 132],  [203,  89, 139],  [ 89,  73,  46],  [185, 234, 136],  [  2, 192, 106],  [176,  63,  69],  [ 52, 181,  69],  [ 83,  39,  91],  [153,  80, 145],  [211, 153, 227],  [ 39,  64,  50],  [213, 114, 243]],
        [[243, 179, 224],  [252, 162, 225],  [ 14,  61, 135],  [ 36, 101,  24],  [ 36,  12,  41],  [ 63, 101,  32],  [ 57, 255,   6],  [142,  76,  17],  [ 22, 188, 131],  [ 49,  84,  62],  [152, 137, 118],  [216, 130, 235],  [187, 146,  33],  [238, 130, 107],  [ 74,  56,  12],  [104,  14, 235]],
        [[190, 107,  56],  [ 41,  50,  38],  [  3, 163, 242],  [ 39, 223, 247],  [160, 225, 184],  [191, 247, 150],  [226,  42,   3],  [100, 205,  22],  [ 65,  36, 209],  [ 46, 179,  62],  [202, 186, 185],  [181, 204, 167],  [224, 101,  97],  [228,  66,  46],  [204,  85, 164],  [ 91, 114, 193]],
        [[185,  91, 201],  [ 97,  82, 235],  [ 26, 175, 166],  [103, 178,  38],  [ 29, 108, 147],  [141,  13,  97],  [211, 198,  52],  [244, 203,  53],  [215,  83, 215],  [ 44, 143, 113],  [ 56,  72, 219],  [ 30,  79,  50],  [ 89, 248, 124],  [243, 191,   7],  [159, 171, 153],  [225,  27, 157]],
        [[237,  54,  47],  [236, 164, 133],  [113,  42, 119],  [ 84, 205,  33],  [ 44, 133, 183],  [134,  84,  91],  [184,  70,  32],  [178, 147, 239],  [ 83,   6,   5],  [194,   3, 129],  [ 79, 142, 186],  [227,  62, 140],  [ 38, 160, 196],  [ 86, 214,  38],  [  1,  23,  48],  [113,  22, 108]],
        [[ 82, 252, 114],  [210,  54,  45],  [ 94, 199,  18],  [  2, 251,  15],  [194, 153,  56],  [113, 222, 157],  [229, 172, 142],  [ 79, 105,  18],  [208,  54,  72],  [ 43, 165,  76],  [102, 118, 100],  [ 63, 221, 191],  [161, 161,  23],  [222, 114,   4],  [246,   4,  77],  [150, 216, 244]],
        [[110,  80,  84],  [ 86, 231, 245],  [ 88, 190,  39],  [155, 218, 125],  [145,  31,  91],  [237, 132,  40],  [190, 119,  18],  [200, 211,  78],  [129,   7,  68],  [ 57, 155,  83],  [124,  28, 202],  [ 41, 201,  13],  [161, 175, 206],  [ 13, 104,   6],  [163, 122, 123],  [182,  53,  13]],
        [[197, 140, 145],  [ 12, 206, 249],  [ 91,  80,  72],  [ 14,  14, 125],  [221, 210, 247],  [175,   5,  16],  [218,  98, 236],  [ 34,  60, 169],  [ 52, 151,   9],  [122,  41, 192],  [126, 126,  62],  [237, 146, 139],  [174, 196,  53],  [ 52, 221, 221],  [  1,  44,  17],  [160, 233, 130]],
        [[ 74, 109,   8],  [ 14, 213,  43],  [130, 157, 255],  [ 40, 230,   0],  [185, 128, 199],  [ 56, 121, 131],  [ 62, 116,  95],  [148,  58, 140],  [ 31, 124,  51],  [  5, 181,   4],  [ 77, 148, 156],  [ 26, 110, 164],  [ 62, 188, 130],  [107, 100,  92],  [ 86,  18, 195],  [165,  20,  84]],
        [[ 14, 134,  15],  [ 93,  74, 126],  [ 12, 238,  98],  [236, 207,  13],  [165, 119,  23],  [111, 244, 107],  [ 57,  75, 231],  [100,  38,  55],  [106, 117,  94],  [194,  12, 239],  [189,  13, 120],  [224, 126, 127],  [145,  15,  59],  [ 58, 255,  52],  [225, 167,  31],  [130, 194, 229]],
        [[117, 161,  26],  [169, 167, 221],  [139, 176,  81],  [207,  95,  81],  [190, 167, 170],  [ 81,  56,  70],  [170, 221,  91],  [ 16, 201, 141],  [ 71, 130, 197],  [177,  88,  88],  [245, 251, 127],  [ 97,  90, 148],  [197,  94,  64],  [ 76, 163, 121],  [102,  10, 209],  [176, 114, 223]],
    ], dtype=np.uint8),
]

class TestQOWIDecoder(unittest.TestCase):

    def test_instantiation(self):
        d = QOWIDecoder()
        self.assertIsInstance(d, QOWIDecoder)

    def test_instantiation_from_zero_image(self):
        encoded_bits = BitStream()
        e = QOWIEncoder()
        e.from_array(TEST_IMAGES[0])
        e.to_bitstream(encoded_bits)
        e.encode()

        d = QOWIDecoder()
        d.from_bitstream(encoded_bits)
        d.decode()
        decoded_image = d.as_array()

        self.assertIsInstance(d, QOWIDecoder)

    def test_round_trip_from_zero_image(self):
        source_image = TEST_IMAGES[0]
        encoded_bits = BitStream()

        e = QOWIEncoder()
        e.from_array(source_image)
        e.to_bitstream(encoded_bits)
        e.encode()

        d = QOWIDecoder()
        d.from_bitstream(encoded_bits)
        d.decode()
        decoded_image = d.as_array()

        self.assertTrue(np.array_equal(decoded_image, source_image))

    def test_round_trip_from_two_image(self):
        source_image = TEST_IMAGES[2]
        encoded_bits = BitStream()

        e = QOWIEncoder()
        e.from_array(source_image)
        e.to_bitstream(encoded_bits)
        e.encode()

        d = QOWIDecoder()
        d.from_bitstream(encoded_bits)
        d.decode()
        decoded_image = d.as_array()

        self.assertTrue(np.array_equal(decoded_image, source_image))


    def test_round_trip_from_three_image(self):
        source_image = TEST_IMAGES[3]
        encoded_bits = BitStream()

        e = QOWIEncoder()
        e.from_array(source_image)
        e.to_bitstream(encoded_bits)
        e.encode()

        d = QOWIDecoder()
        d.from_bitstream(encoded_bits)
        d.decode()
        decoded_image = d.as_array()

        self.assertTrue(np.array_equal(decoded_image, source_image))

if __name__ == '__main__':
    unittest.main()
