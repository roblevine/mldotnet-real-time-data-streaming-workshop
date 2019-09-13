using System;
using Microsoft.ML;

namespace FraudPredictionTrainer
{
    class Program
    {
        private static string DataPath = "data.csv";

        static void Main(string[] args)
        {
            var mlContext = new MLContext(seed: 1);
            
            var data = mlContext.Data.LoadFromTextFile<Transaction>(DataPath, hasHeader: true, separatorChar: ',');

            var testTrainData = mlContext.Data.TrainTestSplit(data);

            var dataProcessingPipeline = mlContext.Transforms.Categorical.OneHotEncoding("type")
                .Append(mlContext.Transforms.Categorical.OneHotHashEncoding("nameDest"))
                .Append(mlContext.Transforms.Concatenate("Features", "type", "nameDest", 
                "amount", "oldbalanceOrg", "oldbalanceDest", "newbalanceOrig", "newbalanceDest"));

            var trainingPipeline = dataProcessingPipeline
                .Append(mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression(labelColumnName: "isFraud"));

            var trainedModel = trainingPipeline.Fit(testTrainData.TrainSet);
        }
    }
}
