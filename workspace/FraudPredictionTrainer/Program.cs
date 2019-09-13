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

        }
    }
}
