import unittest
import genericLib

class genericLib_test(unittest.TestCase):
    def test_parse_isoduration(self):
        """
        Test parse_isoduration

        :return: Nada. Just runs tests.
        """

        strings = ['P1Y1M1DT1H1M1.11S', 'P2Y', 'PT1M', 'PT1M2.22S', 'P-1Y']
        expected = [[1, 1, 1, 1, 1, 1.11], [2, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 1, 2.22],
                    [-1, 0, 0, 0, 0, 0]]

        for s, e in zip(strings, expected):
            got = genericLib.parse_isoduration(s)
            self.assertEqual(e, got)
            # now reverse
            got = genericLib.parse_isoduration(e)
            self.assertEqual(got, s)

        # should fail without a leading P or not a string
        for fail_case in ['12Y', '1Y', '3MT2S']:
            with self.assertRaises(ValueError):
                got = genericLib.parse_isoduration(fail_case)
            #


if __name__ == '__main__':
    unittest.main()
