using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace TaxiFarePrediction
{
    /// <summary>
    /// The TaxiTrip class represents a single taxi trip.
    /// </summary>
    public class TaxiTrip
    {
        [LoadColumn(0)] public string VendorId;
        [LoadColumn(5)] public string RateCode;
        [LoadColumn(3)] public float PassengerCount;
        [LoadColumn(4)] public float TripDistance;
        [LoadColumn(9)] public string PaymentType;
        [LoadColumn(10)] public float FareAmount;
    }

    /// <summary>
    /// The TaxiTripFarePrediction class represents a single far prediction.
    /// </summary>
    public class TaxiTripFarePrediction
    {
        [ColumnName("Score")]
        public float FareAmount;
    }

    /// <summary>
    /// The program class.
    /// </summary>
    class Program
    {
        // file paths to data files
        static readonly string dataPath = Path.Combine(Environment.CurrentDirectory, "yellow_tripdata_2018-12.csv");

        /// <summary>
        /// The main application entry point.
        /// </summary>
        /// <param name="args">The command line arguments.</param>
        static void Main(string[] args)
        {
            // create the machine learning context
            var mlContext = new MLContext();

            // set up the text loader 
            var textLoader = mlContext.Data.CreateTextLoader(
                new TextLoader.Options() 
                {
                    Separators = new[] { ',' },
                    HasHeader = true,
                    Columns = new[] 
                    {
                        new TextLoader.Column("VendorId", DataKind.String, 0),
                        new TextLoader.Column("RateCode", DataKind.String, 5),
                        new TextLoader.Column("PassengerCount", DataKind.Single, 3),
                        new TextLoader.Column("TripDistance", DataKind.Single, 4),
                        new TextLoader.Column("PaymentType", DataKind.String, 9),
                        new TextLoader.Column("FareAmount", DataKind.Single, 10)
                    }
                }
            );

            // load the data 
            Console.Write("Loading training data....");
            var dataView = textLoader.Load(dataPath);
            Console.WriteLine("done");

            // split into a training and test partition
            var partitions = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);

            // set up a learning pipeline
            var pipeline = mlContext.Transforms.CopyColumns(
                inputColumnName:"FareAmount", 
                outputColumnName:"Label")

                // one-hot encode all text features
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("VendorId"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("RateCode"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("PaymentType"))

                // combine all input features into a single column 
                .Append(mlContext.Transforms.Concatenate(
                    "Features", 
                    "VendorId", 
                    "RateCode", 
                    "PassengerCount", 
                    "TripDistance", 
                    "PaymentType"))

                // cache the data to speed up training
                .AppendCacheCheckpoint(mlContext)

                // use the fast tree learner 
                .Append(mlContext.Regression.Trainers.FastTree());

            // train the model
            Console.Write("Training the model....");
            var model = pipeline.Fit(partitions.TrainSet);
            Console.WriteLine("done");

            // get a set of predictions 
            Console.Write("Evaluating the model....");
            var predictions = model.Transform(partitions.TestSet);

            // get regression metrics to score the model
            var metrics = mlContext.Regression.Evaluate(predictions, "Label", "Score");
            Console.WriteLine("done");

            // show the metrics
            Console.WriteLine();
            Console.WriteLine($"Model metrics:");
            Console.WriteLine($"  RMSE:{metrics.RootMeanSquaredError:#.##}");
            Console.WriteLine($"  MSE: {metrics.MeanSquaredError:#.##}");
            Console.WriteLine($"  MAE: {metrics.MeanAbsoluteError:#.##}");
            Console.WriteLine();

            // create a prediction engine for one single prediction
            var predictionFunction = mlContext.Model.CreatePredictionEngine<TaxiTrip, TaxiTripFarePrediction>(model);

            // prep a single taxi trip
            var taxiTripSample = new TaxiTrip()
            {
                VendorId = "2",
                RateCode = "1",
                PassengerCount = 1,
                TripDistance = 3.75f,
                PaymentType = "1",
                FareAmount = 0 // the model will predict the  fare for this trip
            };

            // make the prediction
            var prediction = predictionFunction.Predict(taxiTripSample);

            // sho the prediction
            Console.WriteLine($"Single prediction:");
            Console.WriteLine($"  Predicted fare: {prediction.FareAmount:0.####}");

            Console.ReadLine();
        }

    }
}
