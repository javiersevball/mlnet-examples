using System;
using System.Linq;
using BinaryClassificationApp.Model;

namespace BinaryClassificationApp
{
    /// <summary>
    /// Class that implements the console application.
    /// </summary>
    public class Program
    {
        #region Fields

        /// <summary>
        /// Field that contains the model instance.
        /// </summary>
        private static ReviewSentimentModel _model = new ReviewSentimentModel("Model.mlnet");

        #endregion

        public static void Main(string[] args)
        {
            Console.WriteLine("--- ML-NET BINARY CLASSIFICATION EXAMPLE");

            Console.WriteLine("Training...");
            TrainModel("InputData/yelp_labelled.txt");
        }

        /// <summary>
        /// Method that trains the ML-NET model.
        /// </summary>
        /// <param name="inputPath">The input data path.</param>
        private static void TrainModel(string inputPath)
        {
            _model.Train(inputPath, '\t', false);
        }
    }
}
