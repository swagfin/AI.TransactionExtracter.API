using Microsoft.AspNetCore.Mvc;
using Microsoft.ML;

namespace AI.TransactionExtracter.API.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class ModelController : ControllerBase
    {
        private readonly ILogger<ModelController> _logger;
        private readonly string _dataPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "training-data.tsv");
        private readonly string _modelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "ner-model.zip");

        public ModelController(ILogger<ModelController> logger)
        {
            this._logger = logger;
        }


        [HttpGet("train")]
        public ActionResult<string> Get()
        {
            try
            {

                ModelTrainer.Train(_dataPath, _modelPath);

                return Ok(new
                {
                    Success = true,
                    Trainer = _dataPath,
                    Model = _modelPath
                });
            }
            catch (Exception ex)
            {
                _logger.LogError("Error: {Message}", ex.Message);
                return BadRequest(ex.Message);
            }
        }


        [HttpPost("test")]
        public ActionResult<string> Post([FromBody] string message = "You have received 500 KES in your account. Ref 123456.")
        {

            try
            {
                ArgumentNullException.ThrowIfNull(message);

                string[] testMessages = new[]
                {
                    "You have received 500 KES in your account. Ref 123456.",
                    "M-PESA deposit of 1000 shs received Ref AB12345Z.",
                    "Paid Ksh 750 to your account Ref REF2024A.",
                    "Amount of 1200 Ksh was transferred. Ref TXN444555.",
                    "You got credited with 3000 KES. Reference number REF998877.",
                    "Ksh 200 deposited into your wallet. Ref 111XYZ999.",
                    "Received 2500 KES. Transaction ID TRX345677.",
                    "Sent KES 1500. Reference code XYZABC123.",
                    "Your payment of 1800 has been received Ref REF22001X.",
                    "Received cash of Ksh 600 with Ref REFHELLO001."
                };


                MLContext mlContext = new MLContext();
                // Use in production
                ModelPredictor predictor = new ModelPredictor(_modelPath);
                List<SmsTokenPredictionResult> results = predictor.ExtractEntities(message);

                return Ok(results.Where(x => x.PredictedLabel != "O").ToList());
            }
            catch (Exception ex)
            {
                _logger.LogError("Error: {Message}", ex.Message);
                return BadRequest(ex.Message);
            }
        }

    }
}
