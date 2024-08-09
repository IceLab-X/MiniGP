# Model Comparison and Regression test
- **Model_comparison.py:** A Python script that compares the performance of different GP models on different synthetic datasets including periodic, warped and polynomial. Default models are set as autoGP and its base model vsgp. 
<img src="https://github.com/IceLab-X/Mini-GP/blob/051c6a3a60b955ffdea8a4fd72606353c21bfd49/core/Model_comparison&Regression_test/Model_comparison_autoGP.png" alt="Model Comparison for autoGP and vsgp" width="800"/>
<img src="https://github.com/IceLab-X/Mini-GP/blob/a669f61a364dccbea7e632db07ffb749e3db3713/core/Model_comparison&Regression_test/Model_comparison_warped.png" alt="Model Comparison for autoGP and vsgp" width="800"/>
- **Regression_test.py:** A Python script that tests the accuracy and training speed of different size of training sets. The result is stored in result1.csv and result2.csv.
- **result1.csv:** The result of the regression test for different size of training sets.
<img src="https://github.com/IceLab-X/Mini-GP/blob/a669f61a364dccbea7e632db07ffb749e3db3713/core/Model_comparison&Regression_test/Model_comparison%20Accuracy.PNG" alt="Model Comparison for autoGP and vsgp" width="400"/>
<img src="https://github.com/IceLab-X/Mini-GP/blob/a669f61a364dccbea7e632db07ffb749e3db3713/core/Model_comparison&Regression_test/Model_comprison%20Speed.PNG" alt="Model Comparison for autoGP and vsgp" width="400"/>
- **result2.csv:** The result of the regression test for different size of number of inducing points.
<img src="https://github.com/IceLab-X/Mini-GP/blob/a669f61a364dccbea7e632db07ffb749e3db3713/core/Model_comparison&Regression_test/Model_comprison%20Accuracy2.PNG" alt="Model Comparison for autoGP and vsgp" width="600"/>
<img src="https://github.com/IceLab-X/Mini-GP/blob/a669f61a364dccbea7e632db07ffb749e3db3713/core/Model_comparison&Regression_test/Model_comprison%20Speed2.PNG" alt="Model Comparison for autoGP and vsgp" width="600"/>