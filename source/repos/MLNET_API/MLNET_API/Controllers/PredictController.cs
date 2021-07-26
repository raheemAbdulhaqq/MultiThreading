using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.ML;
using MLNET_API.DataModels;

namespace MLNET_API.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class PredictController : ControllerBase
    {
        private readonly PredictionEnginePool<UserData, UserPrediction> _predictionEnginePool;

        public PredictController(PredictionEnginePool<UserData, UserPrediction> predictionEnginePool)
        {
            _predictionEnginePool = predictionEnginePool;
        }

        [HttpPost]
        public ActionResult<string> Post([FromBody] UserData input)
        {
            if (!ModelState.IsValid)
            {
                return BadRequest();
            }

            UserPrediction prediction = _predictionEnginePool.Predict(modelName: "PlacementModel", example: input);

            string placement = prediction.organization;

            return Ok(placement);
        }
    }
}