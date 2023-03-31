using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using System.Linq;

namespace BinaryClassificationApp.Model
{
    /// <summary>
    /// Class that represents the review sentiment ML.NET model.
    /// </summary>
    public class ReviewSentimentModel
    {
        /// <summary>
        /// Inner class that determines the format of the model input.
        /// </summary>
        public class ModelInput
        {
            [LoadColumn(0)]
            [ColumnName("Review")]
            public string ReviewText { get; set; }

            [LoadColumn(1)]
            [ColumnName("Sentiment")]
            public string Sentiment { get; set; }
        }

        /// <summary>
        /// Inner class that determines the output class for the ML-NET model.
        /// </summary>
        public class ModelOutput
        {
            [ColumnName("ReviewFeatures")]
            public float[] ReviewFeatures { get; set; }

            [ColumnName("SentimentKey")]
            public uint SentimentKey { get; set; }

            [ColumnName(@"Features")]
            public float[] Features { get; set; }

            [ColumnName(@"PredictedLabel")]
            public float PredictedLabel { get; set; }

            [ColumnName(@"Score")]
            public float[] Score { get; set; }
        }

        #region Fields

        /// <summary>
        /// Determines the path to the ML-NET model file.
        /// </summary>
        private readonly string _modelFilePath;

        /// <summary>
        /// Determines the prediction engine object (lazy initialization).
        /// </summary>
        private readonly Lazy<PredictionEngine<ModelInput, ModelOutput>> _predictEngine;

        #endregion

        #region Constructors

        /// <summary>
        /// Initializes a new instance of the <see cref="ReviewSentimentModel" />.
        /// </summary>
        public ReviewSentimentModel(string modelFilePath)
        {
            _modelFilePath = modelFilePath;

            _predictEngine = new Lazy<PredictionEngine<ModelInput, ModelOutput>>(
                () => CreatePredictionEngine(), true);
        }

        #endregion

        #region Methods

        /// <summary>
        /// Train the model with the given input data.
        /// </summary>
        /// <param name="inputDataPath">Path to the input data file.</param>
        /// <param name="separator">The data separator character.</param>
        /// <param name="hasHeader">True if the input file has a header, false otherwise.</param>
        public void Train(string inputDataPath, char separator, bool hasHeader)
        {
            var context = new MLContext();

            // Load data from input path
            var data = context.Data.LoadFromTextFile<ModelInput>(
                inputDataPath, separator, hasHeader);

            // Retrain model
            var pipeline = BuildPipeline(context);
            var model = pipeline.Fit(data);

            // Save the ML-NET model
            using (var fStream = File.Create(_modelFilePath))
            {
                context.Model.Save(model, data.Schema, fStream);
            }
        }

        /// <summary>
        /// Method that predicts scores for all possible labels.
        /// </summary>
        /// <param name="input">The model input.</param>
        /// <returns>TODO</returns>
        public IOrderedEnumerable<KeyValuePair<string, float>> PredictLabels(ModelInput input)
        {
            var engine = _predictEngine.Value;
            var result = engine.Predict(input);

            //return GetSortedScoresWithLabels(result);
            throw new NotImplementedException();
        }

        /// <summary>
        /// Method that builds the pipeline used from model builder.
        /// Maximum entropy multiclass classifier (L-BFGS method).
        /// </summary>
        /// <param name="context">The working ML context.</param>
        /// <returns>The untrained transformer.</returns>
        private IEstimator<ITransformer> BuildPipeline(MLContext context)
        {
            // L-BFGS maximum entropy options (ML-NET CLI)
            var options = new LbfgsMaximumEntropyMulticlassTrainer.Options();
            options.L1Regularization = 0.03125F;
            options.L2Regularization = 0.4058975F;
            options.LabelColumnName = "SentimentKey";
            options.FeatureColumnName = "Features";

            var pipeline = context.Transforms.Text.FeaturizeText("ReviewFeatures", "Review")
                .Append(
                    context.Transforms.Concatenate("Features", new string[] { "ReviewFeatures" }))
                .Append(
                    context.Transforms.Conversion.MapValueToKey("SentimentKey", "Sentiment", addKeyValueAnnotationsAsText: false))
                .Append(
                    context.MulticlassClassification.Trainers.LbfgsMaximumEntropy(options))
                .Append(
                    context.Transforms.Conversion.MapValueToKey("PredictedLabel", "PredictedLabel"));

            return pipeline;
        }

        /// <summary>
        /// Method that creates the prediction engine.
        /// </summary>
        /// <returns>Generated prediction engine object.</returns>
        private PredictionEngine<ModelInput, ModelOutput> CreatePredictionEngine()
        {
            var context = new MLContext();
            var model = context.Model.Load(_modelFilePath, out var _);

            return context.Model.CreatePredictionEngine<ModelInput, ModelOutput>(model);
        }

        /// <summary>
        /// Get the ordered labels.
        /// </summary>
        /// <param name="result">Predicted result to get the labels from.</param>
        /// <returns>The list of labels.</returns>
        private IEnumerable<string> GetLabels(ModelOutput result)
        {
            var schema = _predictEngine.Value.OutputSchema;

            var labelColumn = schema.GetColumnOrNull("SentimentKey");
            if (labelColumn == null)
            {
                throw new Exception(
                    "'SentimentKey' column not found.");
            }

            // Key values contains an ordered array of all possible labels
            var names = new VBuffer<float>();
            labelColumn.Value.GetKeyValues(ref names);

            return names.DenseValues().Select(x => x.ToString());
        }

        #endregion
    }
}
