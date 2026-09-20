from specialist_models.sar_agent import SARAgent

IMAGE_PATH = "/home/rikin/satquery-ai/satquery_sar_test/sample_VV.tif"


agent = SARAgent()

metadata, top_prediction, predictions = agent.predict(
        image_path=IMAGE_PATH,
        #candidate_labels=candidate_labels
    )


print(metadata)
print(top_prediction)
print(predictions)