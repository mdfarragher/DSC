using System;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Trainers;
using Microsoft.ML.Data;

namespace MovieRecommender
{
    /// <summary>
    /// The MovieRating class holds a single movie rating.
    /// </summary>
    public class MovieRating
    {
        [LoadColumn(0)] public float UserID;
        [LoadColumn(1)] public float MovieID;
        [LoadColumn(2)] public float Label;
    }

    /// <summary>
    /// The MovieRatingPrediction class holds a single movie prediction.
    /// </summary>
    public class MovieRatingPrediction
    {
        public float Label;
        public float Score;
    }

    /// <summary>
    /// The main program class.
    /// </summary>
    class Program
    {
        // filenames for training and test data
        private static string trainingDataPath = Path.Combine(Environment.CurrentDirectory, "recommendation-ratings-train.csv");
        private static string testDataPath = Path.Combine(Environment.CurrentDirectory, "recommendation-ratings-test.csv");

        /// <summary>
        /// The program entry point.
        /// </summary>
        /// <param name="args">The command line arguments</param>
        static void Main(string[] args)
        {
            // set up a new machine learning context
            var context = new MLContext();

            // load training and test data
            var trainingDataView = context.Data.LoadFromTextFile<MovieRating>(trainingDataPath, hasHeader: true, separatorChar: ',');
            var testDataView = context.Data.LoadFromTextFile<MovieRating>(testDataPath, hasHeader: true, separatorChar: ',');

            // prepare matrix factorization options
            var options = new MatrixFactorizationTrainer.Options
            {
                MatrixColumnIndexColumnName = "UserIDEncoded",
                MatrixRowIndexColumnName = "MovieIDEncoded", 
                LabelColumnName = "Label",
                NumberOfIterations = 20,
                ApproximationRank = 100
            };

            // set up a training pipeline
            // step 1: map userId and movieId to keys
            var pipeline = context.Transforms.Conversion.MapValueToKey(
                    inputColumnName: "UserID",
                    outputColumnName: "UserIDEncoded")
                .Append(context.Transforms.Conversion.MapValueToKey(
                    inputColumnName: "MovieID",
                    outputColumnName: "MovieIDEncoded")

                // step 2: find recommendations using matrix factorization
                .Append(context.Recommendation().Trainers.MatrixFactorization(options)));

            // train the model
            Console.WriteLine("Training the model...");
            var model = pipeline.Fit(trainingDataView);  
            Console.WriteLine();

            // evaluate the model performance 
            Console.WriteLine("Evaluating the model...");
            var predictions = model.Transform(testDataView);
            var metrics = context.Regression.Evaluate(predictions, labelColumnName: "Label", scoreColumnName: "Score");
            Console.WriteLine($"  RMSE: {metrics.RootMeanSquaredError:#.##}");
            Console.WriteLine($"  MAE:  {metrics.MeanAbsoluteError:#.##}");
            Console.WriteLine($"  MSE:  {metrics.MeanSquaredError:#.##}");
            Console.WriteLine();

            // check if Mark likes 'GoldenEye'
            Console.WriteLine("Calculating the score for Mark liking the movie 'GoldenEye'...");
            var predictionEngine = context.Model.CreatePredictionEngine<MovieRating, MovieRatingPrediction>(model);
            var prediction = predictionEngine.Predict(
                new MovieRating()
                {
                    UserID = 999,
                    MovieID = 10  // GoldenEye
                }
            );
            Console.WriteLine($"  Score: {prediction.Score}");
            Console.WriteLine();

            // find Mark's top 5 movies
            Console.WriteLine("Calculating Mark's top 5 movies...");
            var top5 =  (from m in Movies.All
                         let p = predictionEngine.Predict(
                            new MovieRating()
                            {
                                UserID = 999,
                                MovieID = m.ID
                            })
                         orderby p.Score descending
                         select (MovieId: m.ID, Score: p.Score)).Take(5);
            foreach (var t in top5)
                Console.WriteLine($"  Score:{t.Score}\tMovie: {Movies.Get(t.MovieId)?.Title}");

            Console.ReadLine();
        }
    }
}
