using Microsoft.ML;
using Microsoft.ML.Data;
namespace AI.TransactionExtracter.API
{
    public static class ModelTrainer
    {
        public static void Train(string dataPath, string modelPath)
        {
            var mlContext = new MLContext();

            // Load raw data from TSV
            IDataView data = mlContext.Data.LoadFromTextFile<SmsToken>(
                      path: dataPath,
                      hasHeader: false,
                      separatorChar: '\t');

            // Convert to IEnumerable to normalize labels
            var normalizedData = mlContext.Data
                .CreateEnumerable<SmsToken>(data, reuseRowObject: false)
                .Select(x => new SmsToken
                {
                    Word = x.Word,
                    Label = x.Label?.ToUpperInvariant() ?? "O"
                });

            // Convert back to IDataView after normalization
            IDataView cleanData = mlContext.Data.LoadFromEnumerable(normalizedData);

            // Build pipeline
            var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label", "Label")
                .Append(mlContext.Transforms.Text.FeaturizeText("Features", "Word"))
                .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            // Train
            var model = pipeline.Fit(cleanData);

            // Save the trained model
            mlContext.Model.Save(model, cleanData.Schema, modelPath);

            Console.WriteLine("✅ Model trained and saved to: " + modelPath);
        }
    }
    public class SmsToken
    {
        [LoadColumn(0)]
        public string Word { get; set; }

        [LoadColumn(1)]
        public string Label { get; set; }
    }

}
