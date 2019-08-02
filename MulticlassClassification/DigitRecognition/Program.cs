using System;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using BetterConsoleTables;

namespace Mnist
{
    /// <summary>
    /// The Digit class represents one mnist digit.
    /// </summary>
    class Digit
    {
        [ColumnName("PixelValues")]
        [VectorType(784)]
        public float[] PixelValues;

        [LoadColumn(0)]
        public float Number;
    }

    /// <summary>
    /// The DigitPrediction class represents one digit prediction.
    /// </summary>
    class DigitPrediction
    {
        [ColumnName("Score")]
        public float[] Score;
    }

    /// <summary>
    /// The main program class.
    /// </summary>
    class Program
    {
        // filenames for data set
        private static string trainDataPath = Path.Combine(Environment.CurrentDirectory, "mnist_train.csv");
        private static string testDataPath = Path.Combine(Environment.CurrentDirectory, "mnist_test.csv");

        /// <summary>
        /// The main program entry point.
        /// </summary>
        /// <param name="args">The command line arguments.</param>
        static void Main(string[] args)
        {
            // create a machine learning context
            var context = new MLContext();

            // load data
            Console.WriteLine("Loading data....");
            var columnDef = new TextLoader.Column[]
            {
                new TextLoader.Column(nameof(Digit.PixelValues), DataKind.Single, 1, 784),
                new TextLoader.Column("Number", DataKind.Single, 0)
            };
            var trainDataView = context.Data.LoadFromTextFile(
                path: trainDataPath,
                columns : columnDef,
                hasHeader : true,
                separatorChar : ',');
            var testDataView = context.Data.LoadFromTextFile(
                path: testDataPath,
                columns : columnDef,
                hasHeader : true,
                separatorChar : ',');


            // build a training pipeline
            // step 1: map the number column to a key value and store in the label column
            var pipeline = context.Transforms.Conversion.MapValueToKey(
                outputColumnName: "Label", 
                inputColumnName: "Number", 
                keyOrdinality: ValueToKeyMappingEstimator.KeyOrdinality.ByValue)

                // step 2: concatenate all feature columns
                .Append(context.Transforms.Concatenate(
                    "Features", 
                    nameof(Digit.PixelValues)))
                    
                // step 3: cache data to speed up training                
                .AppendCacheCheckpoint(context)

                // step 4: train the model with SDCA
                .Append(context.MulticlassClassification.Trainers.SdcaMaximumEntropy(
                    labelColumnName: "Label", 
                    featureColumnName: "Features"))

                // step 5: map the label key value back to a number
                .Append(context.Transforms.Conversion.MapKeyToValue(
                    outputColumnName: "Number",
                    inputColumnName: "Label"));

            // train the model
            Console.WriteLine("Training model....");
            var model = pipeline.Fit(trainDataView);

            // use the model to make predictions on the test data
            Console.WriteLine("Evaluating model....");
            var predictions = model.Transform(testDataView);

            // evaluate the predictions
            var metrics = context.MulticlassClassification.Evaluate(
                data: predictions, 
                labelColumnName: "Number", 
                scoreColumnName: "Score");

            // show evaluation metrics
            Console.WriteLine($"Evaluation metrics");
            Console.WriteLine($"    MicroAccuracy:    {metrics.MicroAccuracy:0.###}");
            Console.WriteLine($"    MacroAccuracy:    {metrics.MacroAccuracy:0.###}");
            Console.WriteLine($"    LogLoss:          {metrics.LogLoss:#.###}");
            Console.WriteLine($"    LogLossReduction: {metrics.LogLossReduction:#.###}");
            Console.WriteLine();

            // grab three digits from the test data
            var digits = context.Data.CreateEnumerable<Digit>(testDataView, reuseRowObject: false).ToArray();
            var testDigits = new Digit[] { digits[5], digits[16], digits[28], digits[63], digits[129] };

            // create a prediction engine
            var engine = context.Model.CreatePredictionEngine<Digit, DigitPrediction>(model);

            // set up a table to show the predictions
            var table = new Table(TableConfiguration.Unicode());
            table.AddColumn("Digit");
            for (var i = 0; i < 10; i++)
                table.AddColumn($"P{i}");

            // predict each test digit
            for (var i=0; i < testDigits.Length; i++)
            {
                var prediction = engine.Predict(testDigits[i]);
                table.AddRow(
                    testDigits[i].Number, 
                    prediction.Score[0].ToString("P2"),
                    prediction.Score[1].ToString("P2"),
                    prediction.Score[2].ToString("P2"),
                    prediction.Score[3].ToString("P2"),
                    prediction.Score[4].ToString("P2"),
                    prediction.Score[5].ToString("P2"),
                    prediction.Score[6].ToString("P2"),
                    prediction.Score[7].ToString("P2"),
                    prediction.Score[8].ToString("P2"),
                    prediction.Score[9].ToString("P2"));
            }

            // show results
            Console.WriteLine(table.ToString());
            Console.ReadKey();
        }
    }
}
