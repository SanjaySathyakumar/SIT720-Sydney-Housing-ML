from pathlib import Path
import joblib
import pandas as pd
import gradio as gr

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model.joblib"
model = joblib.load(MODEL_PATH)


def predict_price(suburb, property_type, bedrooms, bathrooms, car_spaces, sale_month):
    bedrooms = int(bedrooms)
    bathrooms = int(bathrooms)
    car_spaces = int(car_spaces)
    sale_month = int(sale_month)
    row = pd.DataFrame([{
        "suburb": suburb,
        "property_type": property_type,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "car_spaces": car_spaces,
        "sale_month": sale_month,
        "total_rooms": bedrooms + bathrooms,
        "amenity_score": bathrooms + car_spaces,
        "is_house": 1 if property_type == "House" else 0,
    }])
    prediction = float(model.predict(row)[0])
    return f"Estimated sale price: AUD ${prediction:,.0f}"


with gr.Blocks(title="Sydney Housing Price Decision Support") as demo:
    gr.Markdown(
        "# Sydney Housing Price Decision Support\n"
        "Enter basic property details to obtain an indicative sale-price estimate from the trained model. "
        "This is a learning prototype, not a professional valuation."
    )
    with gr.Row():
        suburb = gr.Dropdown(["Epping", "Parramatta", "Liverpool"], value="Parramatta", label="Suburb")
        property_type = gr.Dropdown(["Unit", "Apartment", "Townhouse", "House"], value="Unit", label="Property type")
    with gr.Row():
        bedrooms = gr.Slider(1, 6, value=2, step=1, label="Bedrooms")
        bathrooms = gr.Slider(1, 5, value=2, step=1, label="Bathrooms")
        car_spaces = gr.Slider(0, 5, value=1, step=1, label="Car spaces")
        sale_month = gr.Slider(1, 12, value=9, step=1, label="Sale month")
    predict = gr.Button("Predict sale price", variant="primary")
    output = gr.Textbox(label="Prediction")
    predict.click(
        predict_price,
        inputs=[suburb, property_type, bedrooms, bathrooms, car_spaces, sale_month],
        outputs=output,
    )
    gr.Markdown(
        "**Caution:** The model was trained on a small manually collected sample. It does not include full property condition, "
        "renovation quality, exact floor/land area for most records, views, street position or other professional valuation inputs."
    )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=False)
