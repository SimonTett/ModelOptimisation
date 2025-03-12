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

    def test_seconds_to_isodurtion(self):
        # test seconds to isoduration
        # test cases

        test_cases = [
            (3661.11, 'PT1H1M1.110S'),
            (60, 'PT1M'),
            (3600, 'PT1H'),
            (86400, 'P1D'),
            (31536000, 'P365D')
        ]

        for seconds, expected in test_cases:
            got = genericLib.seconds_to_isoduration(seconds)
            self.assertEqual(expected, got)
            # test reversability
            reverse = genericLib.parse_isoduration(got)
            # need to convert to seconds..
            rev_seconds =0.0
            for rev,conv in zip(reverse, [31536000, 2592000, 86400, 3600, 60, 1]):
                rev_seconds += rev * conv

            self.assertEqual(seconds, rev_seconds,msg=f'{got} failed for {seconds} with {reverse}')

        # try -ve seconds which should fail
        with self.assertRaises(ValueError):
            got = genericLib.seconds_to_isoduration(-1.00)







if __name__ == '__main__':
    unittest.main()
