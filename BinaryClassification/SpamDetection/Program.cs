using System;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace SpamDetection
{
    /// <summary>
    /// The SpamInput class contains one single message which may be spam or ham.
    /// </summary>
    public class SpamInput
    {
        [LoadColumn(0)] public string RawLabel { get; set; }
        [LoadColumn(1)] public string Message { get; set; }
    }

    /// <summary>
    /// The SpamPrediction class contains one single spam prediction.
    /// </summary>
    public class SpamPrediction
    {
        [ColumnName("PredictedLabel")] public bool IsSpam { get; set; }
        public float Score { get; set; }
        public float Probability { get; set; }
    }

    /// <summary>
    /// This class describes which input columns we want to transform.
    /// </summary>
    public class FromLabel
    {
        public string RawLabel { get; set; }
    }

    /// <summary>
    /// This class describes what output columns we want to produce.
    /// </summary>
    public class ToLabel
    {
        public bool Label { get; set; }
    }

    /// <summary>
    /// The main program class.
    /// </summary>
    public class Program
    {
        // filenames for data set
        private static string dataPath = Path.Combine(Environment.CurrentDirectory, "spam.tsv");

        /// <summary>
        /// The main program entry point.
        /// </summary>
        /// <param name="args">The command line parameters.</param>
        static void Main(string[] args)
        {
            // set up a machine learning context
            var context = new MLContext();

            // load the spam dataset in memory
            var data = context.Data.LoadFromTextFile<SpamInput>(
                path: dataPath, 
                hasHeader: true, 
                separatorChar: '\t');

            // use 80% for training and 20% for testing
            var partitions = context.Data.TrainTestSplit(
                data, 
                testFraction: 0.2);

            // set up a training pipeline
            // step 1: transform the 'spam' and 'ham' values to true and false
            var pipeline = context.Transforms.CustomMapping<FromLabel, ToLabel>(
                mapAction: (input, output) => { output.Label = input.RawLabel == "spam" ? true : false; }, 
                contractName: "MyLambda")

                // step 2: featureize the input text
                .Append(context.Transforms.Text.FeaturizeText(
                    outputColumnName: "Features", 
                    inputColumnName: nameof(SpamInput.Message)))

                // step 3: use a stochastic dual coordinate ascent learner
                .Append(context.BinaryClassification.Trainers.SdcaLogisticRegression());

            // test the full data set by performing k-fold cross validation
            Console.WriteLine("Performing cross validation...");
            var cvResults = context.BinaryClassification.CrossValidate(
                data: partitions.TrainSet, 
                estimator: pipeline, 
                numberOfFolds: 5);

            // report the results
            foreach (var r in cvResults)
                Console.WriteLine($"  Fold: {r.Fold}, AUC: {r.Metrics.AreaUnderRocCurve}");
            Console.WriteLine($"   Average AUC: {cvResults.Average(r => r.Metrics.AreaUnderRocCurve)}");
            Console.WriteLine();

            // train the model on the training set
            Console.WriteLine("Training the model...");
            var model = pipeline.Fit(partitions.TrainSet);

            // evaluate the model on the test set
            Console.WriteLine("Evaluating the model...");
            var predictions = model.Transform(partitions.TestSet);
            var metrics = context.BinaryClassification.Evaluate(
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
            Console.WriteLine("Predicting spam probabilities for a sample messages...");
            var predictionEngine = context.Model.CreatePredictionEngine<SpamInput, SpamPrediction>(model);

            // create sample messages
            var messages = new SpamInput[] {
                new SpamInput() { Message = "Hi, wanna grab lunch together today?" },
                new SpamInput() { Message = "Win a Nokia, PSP, or €25 every week. Txt YEAHIWANNA now to join" },
                new SpamInput() { Message = "Home in 30 mins. Need anything from store?" },
                new SpamInput() { Message = "CONGRATS U WON LOTERY CLAIM UR 1 MILIONN DOLARS PRIZE" },
            };

            // make the prediction
            var myPredictions = from m in messages
                                select (Message: m.Message, Prediction: predictionEngine.Predict(m));

            // show the results
            foreach (var p in myPredictions)
                Console.WriteLine($"  [{p.Prediction.Probability:P2}] {p.Message}");

            Console.ReadLine();
        }
    }
}
