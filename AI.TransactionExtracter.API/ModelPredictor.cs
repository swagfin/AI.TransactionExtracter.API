using Microsoft.ML;
using Microsoft.ML.Data;

namespace AI.TransactionExtracter.API
{
    public class ModelPredictor
    {
        private readonly MLContext _mlContext;
        private readonly ITransformer _model;
        private readonly PredictionEngine<SmsTokenInput, SmsTokenPrediction> _engine;

        public ModelPredictor(string modelPath)
        {
            _mlContext = new MLContext();
            _model = _mlContext.Model.Load(modelPath, out _);
            _engine = _mlContext.Model.CreatePredictionEngine<SmsTokenInput, SmsTokenPrediction>(_model);
        }

        public List<SmsTokenPredictionResult> ExtractEntities(string message)
        {
            List<string> tokens = Tokenize(message);
            var results = new List<SmsTokenPredictionResult>();

            foreach (var word in tokens)
            {
                var prediction = _engine.Predict(new SmsTokenInput { Word = word });
                results.Add(new SmsTokenPredictionResult
                {
                    Word = word,
                    PredictedLabel = prediction.PredictedLabel
                });
            }

            return results;
        }

        private List<string> Tokenize(string text)
        {
            // Basic whitespace + punctuation tokenizer
            return [.. text.Split(new[] { ' ', ',', '.', ':', ';', '-', '_' }, System.StringSplitOptions.RemoveEmptyEntries)];
        }
    }

    public class SmsTokenInput
    {
        public string Label { get; set; }
        public string Word { get; set; }
    }

    public class SmsTokenPrediction
    {
        [ColumnName("PredictedLabel")]
        public string PredictedLabel { get; set; }
    }

    public class SmsTokenPredictionResult
    {
        public string Word { get; set; }
        public string PredictedLabel { get; set; }
    }

}
