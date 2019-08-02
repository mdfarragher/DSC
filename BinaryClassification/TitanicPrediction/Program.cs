using System;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

namespace TitanicPrediction
{
    /// <summary>
    /// The RawAge class is a helper class for a column transformation.
    /// </summary>
    public class FromAge
    {
        public string RawAge;
    }

    /// <summary>
    /// The ProcessedAge class is a helper class for a column transformation.
    /// </summary>
    public class ToAge
    {
        public string Age;
    }

    /// <summary>
    /// The Passenger class represents one passenger on the Titanic.
    /// </summary>
    public class Passenger
    {
        public bool Label;
        public float Pclass;
        public string Name;
        public string Sex;
        public string RawAge;
        public float SibSp;
        public float Parch;
        public string Ticket;
        public float Fare;
        public string Cabin;
        public string Embarked;
    }

    /// <summary>
    /// The PassengerPrediction class represents one model prediction. 
    /// </summary>
    public class PassengerPrediction
    {
        [ColumnName("PredictedLabel")] public bool Prediction;
        public float Probability;
        public float Score;
    }

    /// <summary>
    /// The main program class.
    /// </summary>
    public class Program
    {
        // filenames for training and test data
        private static string trainingDataPath = Path.Combine(Environment.CurrentDirectory, "train_data.csv");
        private static string testDataPath = Path.Combine(Environment.CurrentDirectory, "test_data.csv");

        /// <summary>
        /// The main application entry point.
        /// </summary>
        /// <param name="args">The command line arguments.</param>
        public static void Main(string[] args)
        {
            // set up a machine learning context
            var mlContext = new MLContext();

            // set up a text loader
            var textLoader = mlContext.Data.CreateTextLoader(
                new TextLoader.Options() 
                {
                    Separators = new[] { ',' },
                    HasHeader = true,
                    AllowQuoting = true,
                    Columns = new[] 
                    {
                        new TextLoader.Column("Label", DataKind.Boolean, 1),
                        new TextLoader.Column("Pclass", DataKind.Single, 2),
                        new TextLoader.Column("Name", DataKind.String, 3),
                        new TextLoader.Column("Sex", DataKind.String, 4),
                        new TextLoader.Column("RawAge", DataKind.String, 5),  // <-- not a float!
                        new TextLoader.Column("SibSp", DataKind.Single, 6),
                        new TextLoader.Column("Parch", DataKind.Single, 7),
                        new TextLoader.Column("Ticket", DataKind.String, 8),
                        new TextLoader.Column("Fare", DataKind.Single, 9),
                        new TextLoader.Column("Cabin", DataKind.String, 10),
                        new TextLoader.Column("Embarked", DataKind.String, 11)
                    }
                }
            );

            // load training and test data
            Console.WriteLine("Loading data...");
            var trainingDataView = textLoader.Load(trainingDataPath);
            var testDataView = textLoader.Load(testDataPath);

            // set up a training pipeline
            // step 1: drop the name, cabin, and ticket columns
            var pipeline = mlContext.Transforms.DropColumns("Name", "Cabin", "Ticket")

                // step 2: replace missing ages with '?'
                .Append(mlContext.Transforms.CustomMapping<FromAge, ToAge>(
                    (inp, outp) => { outp.Age = string.IsNullOrEmpty(inp.RawAge) ? "?" : inp.RawAge; },
                    "AgeMapping"
                ))

                // step 3: convert string ages to floats
                .Append(mlContext.Transforms.Conversion.ConvertType(
                    "Age",
                    outputKind: DataKind.Single
                ))

                // step 4: replace missing age values with the mean age
                .Append(mlContext.Transforms.ReplaceMissingValues(
                    "Age",
                    replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean))
 
                // step 5: replace sex and embarked columns with one-hot encoded vectors
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("Sex"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("Embarked"))
 
                // step 6: concatenate everything into a single feature column 
                .Append(mlContext.Transforms.Concatenate(
                    "Features", 
                    "Age",
                    "Pclass", 
                    "SibSp",
                    "Parch",
                    "Sex",
                    "Embarked"))
 
                // step 7: use a fasttree trainer
                .Append(mlContext.BinaryClassification.Trainers.FastTree(
                    labelColumnName: "Label", 
                    featureColumnName: "Features"));

            // show the first 10 data records in the pipeline
            // WritePipeline(mlContext, trainingDataView, pipeline, 10);

            // train the model
            Console.WriteLine("Training model...");
            var trainedModel = pipeline.Fit(trainingDataView);

            // make predictions for the test data set
            Console.WriteLine("Evaluating model...");
            var predictions = trainedModel.Transform(testDataView);

            // compare the predictions with the ground truth
            var metrics = mlContext.BinaryClassification.Evaluate(
                data: predictions, 
                labelColumnName: "Label", 
                scoreColumnName: "Score");

            // report the results
            Console.WriteLine($"  Accuracy:          {metrics.Accuracy:P2}");
            Console.WriteLine($"  Auc:               {metrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"  Auprc:             {metrics.AreaUnderPrecisionRecallCurve:P2}");
            Console.WriteLine($"  F1Score:           {metrics.F1Score:P2}");
            Console.WriteLine($"  LogLoss:           {metrics.LogLoss:0.##}");
            Console.WriteLine($"  LogLossReduction:  {metrics.LogLossReduction:0.##}");
            Console.WriteLine($"  PositivePrecision: {metrics.PositivePrecision:0.##}");
            Console.WriteLine($"  PositiveRecall:    {metrics.PositiveRecall:0.##}");
            Console.WriteLine($"  NegativePrecision: {metrics.NegativePrecision:0.##}");
            Console.WriteLine($"  NegativeRecall:    {metrics.NegativeRecall:0.##}");
            Console.WriteLine();

            // set up a prediction engine
            Console.WriteLine("Making a prediction...");
            var predictionEngine = mlContext.Model.CreatePredictionEngine<Passenger, PassengerPrediction>(trainedModel);

            // create a sample record
            var passenger = new Passenger()
            { 
                Pclass = 1,
                Name = "Mark Farragher",
                Sex = "male",
                RawAge = "48",
                SibSp = 0,
                Parch = 0,
                Fare = 70,
                Embarked = "S"
            };

            // make the prediction
            var prediction = predictionEngine.Predict(passenger);

            // report the results
            Console.WriteLine($"Passenger:   {passenger.Name} ");
            Console.WriteLine($"Prediction:  {(prediction.Prediction ? "survived" : "perished" )} ");
            Console.WriteLine($"Probability: {prediction.Probability} ");            
        }

    }
}
