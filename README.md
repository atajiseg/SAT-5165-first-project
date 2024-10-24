This project aimed to analyze a cardiovascular dataset containing 6,607 records and 20 features related to health and lifestyle factors using Python and Apache Spark. The analysis focused on exploring relationships among variables affecting cardiovascular health and uncovering insights that could guide future health interventions.
The methodology involved loading the dataset and initializing a Spark session to facilitate distributed computing. Data cleaning included handling missing values and selecting relevant features for analysis. A Vector Assembler was used to create feature vectors, which were then standardized using a StandardScaler to ensure equal contribution of all features.
Correlation analysis was performed to identify significant relationships among predictors, particularly emphasizing the importance of blood pressure measurements. Visualizations, including histograms, bar plots, and a heatmap of the correlation matrix, provided a comprehensive overview of the data distribution and relationships.
The findings indicated important correlations between blood pressure and cardiovascular outcomes, highlighting potential predictors for further investigation. Overall, the project showcased the effectiveness of combining Python and Spark for data preprocessing and exploratory analysis, providing valuable insights into factors influencing cardiovascular health.


The analysis yielded several key insights:

The correlation matrix revealed significant relationships between blood pressure measurements (ap_hi and ap_lo) and the target variable (cardio), indicating their importance as predictors of cardiovascular health.
The visualizations highlighted the distribution of key predictors and the presence of outliers, which may influence the overall analysis and subsequent modeling efforts.
Conclusion
This project effectively demonstrates the application of Apache Spark and Python for cardiovascular data analysis. The combination of data preprocessing, exploratory data analysis, and correlation analysis provides a comprehensive approach to understanding factors that impact cardiovascular health. The findings suggest important relationships that could inform future health interventions and predictive modeling, emphasizing the value of utilizing distributed computing for large datasets.

