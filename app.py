from flask import Flask, request, jsonify, render_template
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import  pytesseract
from PIL import Image
import os
from decouple import config

app = Flask(__name__)


# Load Pinecone API Key from environment variables
PINECONE_API_KEY = config("PINECONE_API_KEY")
pc = Pinecone(
        api_key=PINECONE_API_KEY
    )

INDEX_NAME = "product-recommendation"
if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )

index = pc.Index(INDEX_NAME)

# Initialize text embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

@app.route('/product-recommendation', methods=['POST'])
def product_recommendation():
    """
    Endpoint for product recommendations based on natural language queries.
    Input: Form data containing 'query' (string).
    Output: JSON with 'products' (array of objects) and 'response' (string).
    """
    data = request.get_json()
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400

    # Convert query to vector embedding
    query_vector = model.encode(query).tolist()

    # Search in Pinecone (top 5 matches)
    search_results = index.query(vector=query_vector, top_k=5, include_metadata=True)

    # Format response
    recommendations = [
        {
            "StockCode": match["id"],
            "Description": match["metadata"]["Description"],
            "UnitPrice": match["metadata"].get("UnitPrice", "N/A"),
            "Score": match["score"]
        }
        for match in search_results["matches"]
    ]

    return jsonify({"query": query, "recommendations": recommendations})

@app.route('/ocr-query', methods=['POST'])
def ocr_query():
    """
    Endpoint to process handwritten queries extracted from uploaded images.
    Input: Form data containing 'image_data' (file, base64-encoded image or direct file upload).
    Output: JSON with 'products' (array of objects) and 'response' (string).
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image = request.files["image"]
    img = Image.open(image)

    # Extract text from image
    extracted_text = pytesseract.image_to_string(img).strip()

    if not extracted_text:
        return jsonify({"error": "No readable text found"}), 400

    return jsonify({"extracted_text": extracted_text})

# Preprocess image for CNN model
def preprocess_image(image):
    img = Image.open(image).convert("RGB")
    img = img.resize((128, 128))  # Resize for CNN model
    img_array = np.array(img) / 255.0  # Normalize
    return np.expand_dims(img_array, axis=0)

# Predict product class
def predict_product(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction)  # Get class ID
    return predicted_class

# Extract feature vector for Pinecone search
def extract_features(image):
    processed_image = preprocess_image(image)
    feature_vector = model.predict(processed_image)[0]  # Extract feature vector
    return feature_vector.tolist()

# Query Pinecone for similar products
def search_similar_products(feature_vector):
    search_results = index.query(vector=feature_vector, top_k=5, include_metadata=True)
    recommendations = [
        {
            "StockCode": match["id"],
            "Description": match["metadata"]["Description"],
            "UnitPrice": match["metadata"].get("UnitPrice", "N/A"),
            "Score": match["score"]
        }
        for match in search_results["matches"]
    ]
    return recommendations

@app.route('/image-product-search', methods=['POST'])
def image_product_search():
    """
    Endpoint to identify and suggest products from uploaded product images.
    Input: Form data containing 'product_image' (file).
    Output: JSON with 'products' (array of objects) and 'response' (string).
    """
    if "product_image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image = request.files["product_image"]

    # Predict product category
    class_id = predict_product(image)

    # Extract features and search for similar products
    feature_vector = extract_features(image)
    similar_products = search_similar_products(feature_vector)

    response_text = f"The product belongs to category {class_id}. Here are similar products."

    return jsonify({
        "predicted_class": int(class_id),
        "similar_products": similar_products,
        "response": response_text
    })


@app.route('/sample_response', methods=['GET'])
def sample_response():
    """
    Endpoint to return a sample JSON response for the API.
    Output: JSON with 'products' (array of objects) and 'response' (string).
    """
    return render_template('sample_response.html')

if __name__ == '__main__':
    app.run(debug=True)
