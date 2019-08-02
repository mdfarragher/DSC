using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Heart
{
    /// <summary>
    /// The HeartData record holds one single heart data record.
    /// </summary>
    public class HeartData 
    {
        [LoadColumn(0)] public float Age { get; set; }
        [LoadColumn(1)] public float Sex { get; set; }
        [LoadColumn(2)] public float Cp { get; set; }
        [LoadColumn(3)] public float TrestBps { get; set; }
        [LoadColumn(4)] public float Chol { get; set; }
        [LoadColumn(5)] public float Fbs { get; set; }
        [LoadColumn(6)] public float RestEcg { get; set; }
        [LoadColumn(7)] public float Thalac { get; set; }
        [LoadColumn(8)] public float Exang { get; set; }
        [LoadColumn(9)] public float OldPeak { get; set; }
        [LoadColumn(10)] public float Slope { get; set; }
        [LoadColumn(11)] public float Ca { get; set; }
        [LoadColumn(12)] public float Thal { get; set; }
        [LoadColumn(13)] public int RawLabel { get; set; }
    }

    /// <summary>
    /// The HeartPrediction class contains a single heart data prediction.
    /// </summary>
    public class HeartPrediction
    {
        [ColumnName("PredictedLabel")] public bool Prediction;
        public float Probability;
        public float Score;
    }

    /// <summary>
    /// The FromLabel class is a helper class for a column transformation.
    /// </summary>
    public class FromLabel
    {
        public int RawLabel;
    }

    /// <summary>
    /// The ToLabel class is a helper class for a column transformation.
    /// </summary>
    public class ToLabel
    {
        public bool Label;
    }


    /// <summary>
    /// The application class.
    /// </summary>
    public class Program
    {
        // filenames for training and test data
        private static string dataPath = Path.Combine(Environment.CurrentDirectory, "processed.cleveland.data.csv");

        /// <summary>
        /// The main applicaton entry point.
        /// </summary>
        /// <param name="args">The command line arguments.</param>
        public static void Main(string[] args)
        {
            // set up a machine learning context
            var context = new MLContext();

            // load training and test data
            Console.WriteLine("Loading data...");
            var data = context.Data.LoadFromTextFile<HeartData>(dataPath, hasHeader: false, separatorChar: ',');

            // split the data into a training and test partition
            var partitions = context.Data.TrainTestSplit(data, testFraction: 0.2);

            // set up a training pipeline
            // step 1: convert the label value to a boolean
            var pipeline = context.Transforms.CustomMapping<FromLabel, ToLabel>(
                    (input, output) => { output.Label = input.RawLabel > 0; },
                    "LabelMapping"
                )
            
                // step 2: concatenate all feature columns
                .Append(context.Transforms.Concatenate(
                "Features", 
                "Age", 
                "Sex", 
                "Cp", 
                "TrestBps",
                "Chol", 
                "Fbs", 
                "RestEcg", 
                "Thalac", 
                "Exang", 
                "OldPeak", 
                "Slope", 
                "Ca", 
                "Thal"))

                // step 3: set up a fast tree learner
                .Append(context.BinaryClassification.Trainers.FastTree(
                    labelColumnName: "Label", 
                    featureColumnName: "Features"));

            // train the model
            Console.WriteLine("Training model...");
            var model = pipeline.Fit(partitions.TrainSet);

            // make predictions for the test data set
            Console.WriteLine("Evaluating model...");
            var predictions = model.Transform(partitions.TestSet);

            // compare the predictions with the ground truth
            var metrics = context.BinaryClassification.Evaluate(
                data: predictions, 
                labelColumnName: "Label", 
                scoreColumnName: "Score");

            // report the results
            Console.WriteLine($"  Accuracy:          {metrics.Accuracy}");
            Console.WriteLine($"  Auc:               {metrics.AreaUnderRocCurve}");
            Console.WriteLine($"  Auprc:             {metrics.AreaUnderPrecisionRecallCurve}");
            Console.WriteLine($"  F1Score:           {metrics.F1Score}");
            Console.WriteLine($"  LogLoss:           {metrics.LogLoss}");
            Console.WriteLine($"  LogLossReduction:  {metrics.LogLossReduction}");
            Console.WriteLine($"  PositivePrecision: {metrics.PositivePrecision}");
            Console.WriteLine($"  PositiveRecall:    {metrics.PositiveRecall}");
            Console.WriteLine($"  NegativePrecision: {metrics.NegativePrecision}");
            Console.WriteLine($"  NegativeRecall:    {metrics.NegativeRecall}");
            Console.WriteLine();

            // set up a prediction engine
            Console.WriteLine("Making a prediction for a sample patient...");
            var predictionEngine = context.Model.CreatePredictionEngine<HeartData, HeartPrediction>(model);

            // create a sample patient
            var heartData = new HeartData()
            { 
                Age = 36.0f,
                Sex = 1.0f,
                Cp = 4.0f,
                TrestBps = 145.0f,
                Chol = 210.0f,
                Fbs = 0.0f,
                RestEcg = 2.0f,
                Thalac = 148.0f,
                Exang = 1.0f,
                OldPeak = 1.9f,
                Slope = 2.0f,
                Ca = 1.0f,
                Thal = 7.0f,
            };

            // make the prediction
            var prediction = predictionEngine.Predict(heartData);

            // report the results
            Console.WriteLine($"  Age: {heartData.Age} ");
            Console.WriteLine($"  Sex: {heartData.Sex} ");
            Console.WriteLine($"  Cp: {heartData.Cp} ");
            Console.WriteLine($"  TrestBps: {heartData.TrestBps} ");
            Console.WriteLine($"  Chol: {heartData.Chol} ");
            Console.WriteLine($"  Fbs: {heartData.Fbs} ");
            Console.WriteLine($"  RestEcg: {heartData.RestEcg} ");
            Console.WriteLine($"  Thalac: {heartData.Thalac} ");
            Console.WriteLine($"  Exang: {heartData.Exang} ");
            Console.WriteLine($"  OldPeak: {heartData.OldPeak} ");
            Console.WriteLine($"  Slope: {heartData.Slope} ");
            Console.WriteLine($"  Ca: {heartData.Ca} ");
            Console.WriteLine($"  Thal: {heartData.Thal} ");
            Console.WriteLine();
            Console.WriteLine($"Prediction: {(prediction.Prediction ? "Elevated heart disease risk" : "Normal heart disease risk" )} ");
            Console.WriteLine($"Probability: {prediction.Probability:P2} ");

            Console.ReadLine();
        }
    }
}
