using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Linq;

// CS0649 compiler warning is disabled because some fields are only 
// assigned to dynamically by ML.NET at runtime
#pragma warning disable CS0649

namespace MyApp
{
    /// <summary>
    /// The application class.
    /// </summary>
    class Program
    {
        /// <summary>
        /// A data transfer class that holds a single iris flower.
        /// </summary>
        public class IrisData
        {
            [LoadColumn(0)]
            public float SepalLength;

            [LoadColumn(1)]
            public float SepalWidth;

            [LoadColumn(2)]
            public float PetalLength;

            [LoadColumn(3)]
            public float PetalWidth;

            [LoadColumn(4)]
            public string Label;
        }

        /// <summary>
        /// A prediction class that holds a single model prediction.
        /// </summary>
        public class IrisPrediction
        {
            [ColumnName("PredictedLabel")]
            public uint ClusterID;

            [ColumnName("Score")]
            public float[] Score;
        }

        /// <summary>
        /// The main application entry point.
        /// </summary>
        /// <param name="args"The command line arguments></param>
        static void Main(string[] args)
        {
            var mlContext = new MLContext();

            // read the iris flower data from a text file
            var data = mlContext.Data.LoadFromTextFile<IrisData>(
                path: "iris-data.csv", 
                hasHeader: false, 
                separatorChar: ',');

            // split the data into a training and testing partition
            var partitions = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);

            // set up a learning pipeline
            // step 1: concatenate features into a single column
            var pipeline = mlContext.Transforms.Concatenate(
                    "Features", 
                    "SepalLength", 
                    "SepalWidth", 
                    "PetalLength", 
                    "PetalWidth")

                // step 2: use k-means clustering to find the iris types
                .Append(mlContext.Clustering.Trainers.KMeans(
                    featureColumnName: "Features",
                    numberOfClusters: 3));

            // train the model on the data file
            Console.WriteLine("Start training model....");
            var model = pipeline.Fit(partitions.TrainSet);
            Console.WriteLine("Model training complete!");

            // evaluate the model
            Console.WriteLine("Evaluating model:");
            var predictions = model.Transform(partitions.TestSet);
            var metrics = mlContext.Clustering.Evaluate(
                predictions, 
                scoreColumnName: "Score", 
                featureColumnName: "Features");
            Console.WriteLine($"   Average distance:       {metrics.AverageDistance}");
            Console.WriteLine($"   Davies Bould index:     {metrics.DaviesBouldinIndex}");

            // show predictions for a couple of flowers
            Console.WriteLine("Predicting 3 flowers from the test set....");
            var flowers = mlContext.Data.CreateEnumerable<IrisData>(partitions.TestSet, reuseRowObject: false).ToArray();
            var flowerPredictions = mlContext.Data.CreateEnumerable<IrisPrediction>(predictions, reuseRowObject: false).ToArray();
            foreach (var i in new int[] { 0, 10, 20 })
            {
                Console.WriteLine($"   Flower: {flowers[i].Label}, prediction: {flowerPredictions[i].ClusterID}");
            }
            Console.ReadLine();
        }
    }
}