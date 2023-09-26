import unittest
from assignment_code import Formula

class MyTestCase(unittest.TestCase):
    """To test whether the functions work properly"""
    def test_subtraction(self):
        #The formula class is called here
        self.assertEqual(Formula.subtraction(self,7,5),2)
        self.assertEqual(Formula.subtraction(self,10,3),7)

    def test_bad(self):
        #Here, it is observed whether a error will be obtained in case the results are not accurate enough
        data = "letters"
        fact ='alpha'
        #strings are used to raise error consciously
        with self.assertRaises(TypeError):
            res = Formula.subtraction(self,data,fact)



if __name__ == '__main__':
    unittest.main()
