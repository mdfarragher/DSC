using System;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using BetterConsoleTables;
using PLplot;

namespace LoadingData
{
    /// <summary>
    /// The HouseBlockData class holds one single housing block data record.
    /// </summary>
    public class HouseBlockData
    {
        [LoadColumn(0)] public float Longitude { get; set; }
        [LoadColumn(1)] public float Latitude { get; set; }
        [LoadColumn(2)] public float HousingMedianAge { get; set; }
        [LoadColumn(3)] public float TotalRooms { get; set; }
        [LoadColumn(4)] public float TotalBedrooms { get; set; }
        [LoadColumn(5)] public float Population { get; set; }
        [LoadColumn(6)] public float Households { get; set; }
        [LoadColumn(7)] public float MedianIncome { get; set; }
        [LoadColumn(8)] public float MedianHouseValue { get; set; }
    }

    /// <summary>
    /// The ToMedianHouseValue class is used in a column data conversion.
    /// </summary>
    public class ToMedianHouseValue
    {
        public float NormalizedMedianHouseValue { get; set; }
    }

    /// <summary>
    /// The ToRoomsPerPerson class is used in a column data conversion.
    /// </summary>
    public class ToRoomsPerPerson
    {
        public float RoomsPerPerson { get; set; }
    }

    /// <summary>
    /// The FromLocation class is used in a column data conversion.
    /// </summary>
    public class FromLocation
    {
        public float[] EncodedLongitude { get; set; }
        public float[] EncodedLatitude { get; set; }
    }

    /// <summary>
    /// The ToLocation class is used in a column data conversion.
    /// </summary>
    public class ToLocation
    {
        public float[] Location { get; set; }
    }

    /// <summary>
    /// The main program class.
    /// </summary>
    class Program
    {
        // filename for data set
        private static string dataPath = Path.Combine(Environment.CurrentDirectory, "california_housing.csv");
        
        /// <summary>
        /// Helper method to write the machine learning pipeline to the console.
        /// </summary>
        /// <param name="preview">The data preview to write.</param>
        /// <param name="numberOfRows">The maximum number of rows to write.</param>
        public static void WritePreview(DataDebuggerPreview preview, int numberOfRows = 10)
        {
            // set up a console table
            var table = new Table(
                TableConfiguration.Unicode(),
                (from c in preview.ColumnView select c.Column.Name).ToArray());

            // fill the table with results
            foreach (var row in preview.RowView)
            {
                table.AddRow((from c in row.Values 
                              select c.Value is VBuffer<float> ? "<vector>" : c.Value
                            ).ToArray());
            }

            // write the table
            Console.WriteLine(table.ToString());
        }

        /// <summary>
        /// Helper method to write a single column preview to the console.
        /// </summary>
        /// <param name="preview">The data preview to write.</param>
        /// <param name="column">The name of the column to write.</param>
        /// <param name="numberOfRows">The maximum number of rows to write.</param>
        public static void WritePreviewColumn(DataDebuggerPreview preview, string column, int numberOfRows = 10)
        {
            // set up a console table
            var table = new Table(TableConfiguration.Unicode(), column);

            // fill the table with results
            foreach (var row in preview.RowView)
            {
                foreach (var col in row.Values)
                {
                    if (col.Key == column)
                    {   
                        var vector = (VBuffer<float>)col.Value;
                        table.AddRow(string.Concat(vector.DenseValues()));
                    }
                }
            }

            // write the table
            Console.WriteLine(table.ToString());
        }

        /// <summary>
        /// The main program entry point.
        /// </summary>
        /// <param name="args">The command line arguments</param>
        static void Main(string[] args)
        {
            // create the machine learning context
            var context = new MLContext();

            // load the dataset
            Console.WriteLine("Loading data...");
            var data = context.Data.LoadFromTextFile<HouseBlockData>(
                path: dataPath, 
                hasHeader:true, 
                separatorChar: ',');

            // keep only records with a median house value < 500,000
            data = context.Data.FilterRowsByColumn(
                data,
                "MedianHouseValue",
                upperBound: 499_999
            );

            // get an array of housing data
            var houses = context.Data.CreateEnumerable<HouseBlockData>(data, reuseRowObject: false).ToArray();

            // plot median house value by longitude
            var pl = new PLStream();
            pl.sdev("pngcairo");                // png rendering
            pl.sfnam("data.png");               // output filename
            pl.spal0("cmap0_alternate.pal");    // alternate color palette
            pl.init();
            pl.env(
                0, 10,                          // x-axis range
                0, 600000,                      // y-axis range
                AxesScale.Independent,          // scale x and y independently
                AxisBox.BoxTicksLabelsAxes);    // draw box, ticks, and num ticks
            pl.lab(
                "Median Income",                // x-axis label
                "Median House Value",           // y-axis label
                "House value by longitude");    // plot title
            pl.sym(
                houses.Select(h => (double)h.MedianIncome).ToArray(),
                houses.Select(h => (double)h.MedianHouseValue).ToArray(),
                (char)218
            );
            pl.eop();

            // build a data loading pipeline
            // step 1: divide the median house value by 1000
            var pipeline = context.Transforms.CustomMapping<HouseBlockData, ToMedianHouseValue>(
                (input, output) => { output.NormalizedMedianHouseValue = input.MedianHouseValue / 1000; },
                contractName: "MedianHouseValue"
            );

            // get a 10-record preview of the transformed data
            // var model = pipeline.Fit(data);
            // var transformedData = model.Transform(data);
            // var preview = transformedData.Preview(maxRows: 10);

            // show the preview
            // WritePreview(preview);

            // step 2: bin the longitude
            var pipeline2 = pipeline.Append(context.Transforms.NormalizeBinning(
                    inputColumnName: "Longitude",
                    outputColumnName: "BinnedLongitude",
                    maximumBinCount: 10
                ))
        
                // step 3: bin the latitude
                .Append(context.Transforms.NormalizeBinning(
                    inputColumnName: "Latitude",
                    outputColumnName: "BinnedLatitude",
                    maximumBinCount: 10
                ))

                // step 4: one-hot encode the longitude
                .Append(context.Transforms.Categorical.OneHotEncoding(
                    inputColumnName: "BinnedLongitude",
                    outputColumnName: "EncodedLongitude"
                ))

                // step 5: one-hot encode the latitude
                .Append(context.Transforms.Categorical.OneHotEncoding(
                    inputColumnName: "BinnedLatitude",
                    outputColumnName: "EncodedLatitude"
                ));

            // step 6: cross the two one-hot encoded columns
            var pipeline3 = pipeline2.Append(context.Transforms.CustomMapping<FromLocation, ToLocation>(
                    (input, output) => { 
                        output.Location = new float[input.EncodedLongitude.Length * input.EncodedLatitude.Length];
                        var index = 0;
                        for (var i = 0; i < input.EncodedLongitude.Length; i++)
                            for (var j = 0; j < input.EncodedLatitude.Length; j++)
                                output.Location[index++] = input.EncodedLongitude[i] * input.EncodedLatitude[j];
                    },
                    contractName: "Location"
                ))

                // step 7: remove all the columns we don't need anymore
                .Append(context.Transforms.DropColumns(
                    "MedianHouseValue",
                    "Longitude",
                    "Latitude",
                    "BinnedLongitude",
                    "BinnedLatitude",
                    "EncodedLongitude",
                    "EncodedLatitude"
                ));

            // get a 10-record preview of the transformed data
            var model = pipeline3.Fit(data);
            var transformedData = model.Transform(data);
            var preview = transformedData.Preview(maxRows: 10);

            // show the location vector
            //WritePreview(preview);
            WritePreviewColumn(preview, "Location");
        }
    }

}