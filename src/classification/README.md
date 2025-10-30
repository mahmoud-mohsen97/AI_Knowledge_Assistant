# Feedback Classification Module

This module provides customer feedback classification using a multi-task transformer model.

## Overview

The feedback classifier uses a single transformer model (XLM-RoBERTa) with two classification heads:
- **Level 1**: Main categories (Technical, Payment, Claims)
- **Level 2**: Subcategories (Login, App_Performance, Refund, Limit, Installment, Coverage, Appeal, Preexisting)

## Training the Model

Before using the classifier, you need to train the model:

1. Open the training notebook:
   ```bash
   jupyter notebook notebooks/feedback_classification_multitask.ipynb
   ```

2. Run all cells to train the model. The model will be saved to:
   - `notebooks/models/multitask/best_model/`
   - `notebooks/label_mappings_multitask.json`

**Note**: Training requires approximately 4-5GB RAM. The notebook includes memory optimization settings for systems with limited RAM.

## API Usage

### Endpoint: `POST /classify`

Classify customer feedback text into categories.

#### Request

```json
{
  "text": "I can't log in! OTP doesn't arrive!",
  "language": "en"
}
```

**Parameters:**
- `text` (string, required): The feedback text to classify
- `language` (string, optional): Language code - `"en"` or `"ar"`. Default: `"en"`

#### Response

```json
{
  "text": "I can't log in! OTP doesn't arrive!",
  "language": "en",
  "level1": "Technical",
  "level1_confidence": 0.9672,
  "level2": "Login",
  "level2_confidence": 0.8687
}
```

**Fields:**
- `text`: Original feedback text
- `language`: Language code
- `level1`: Predicted main category
- `level1_confidence`: Confidence score (0-1) for level 1 prediction
- `level2`: Predicted subcategory
- `level2_confidence`: Confidence score (0-1) for level 2 prediction

### Example Usage

#### Using cURL

```bash
# English feedback
curl -X POST "http://localhost:8000/classify" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "I can'\''t log in! OTP doesn'\''t arrive!",
    "language": "en"
  }'

# Arabic feedback
curl -X POST "http://localhost:8000/classify" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "متى سأستلم المبلغ المسترد؟",
    "language": "ar"
  }'
```

#### Using Python

```python
import requests

# Classify English feedback
response = requests.post(
    "http://localhost:8000/classify",
    json={
        "text": "I can't log in! OTP doesn't arrive!",
        "language": "en"
    }
)

result = response.json()
print(f"Level 1: {result['level1']} (confidence: {result['level1_confidence']:.2%})")
print(f"Level 2: {result['level2']} (confidence: {result['level2_confidence']:.2%})")
```

#### Using JavaScript/Fetch

```javascript
const classifyFeedback = async (text, language = 'en') => {
  const response = await fetch('http://localhost:8000/classify', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ text, language }),
  });
  
  return await response.json();
};

// Usage
const result = await classifyFeedback("I can't log in! OTP doesn't arrive!");
console.log(`Category: ${result.level1} > ${result.level2}`);
```

## Categories

### Level 1 (Main Categories)
1. **Technical** - Technical issues with the app or login
2. **Payment** - Payment, refund, and financial matters
3. **Claims** - Insurance claims and coverage

### Level 2 (Subcategories)
1. **Login** - Login and authentication issues
2. **App_Performance** - App crashes, freezes, or performance problems
3. **Refund** - Refund requests and status
4. **Limit** - Coverage limit inquiries
5. **Installment** - Payment installment questions
6. **Coverage** - Coverage details and eligibility
7. **Appeal** - Claim appeal process
8. **Preexisting** - Pre-existing condition questions

## Error Handling

### 503 Service Unavailable
The model hasn't been trained yet. Train the model using the notebook first.

```json
{
  "detail": "Classifier model not available. Please train the model first."
}
```

### 400 Bad Request
Invalid input (e.g., empty text or invalid language code).

```json
{
  "detail": "Language must be 'en' or 'ar'"
}
```

### 500 Internal Server Error
Unexpected error during classification.

## Model Details

- **Base Model**: XLM-RoBERTa (multilingual)
- **Architecture**: Multi-task learning with shared encoder
- **Languages**: English and Arabic
- **Size**: ~560 MB
- **Training Time**: ~10-15 minutes (CPU)
- **Inference Time**: ~100-200ms per request

## Health Check

Check if the classifier is loaded:

```bash
curl http://localhost:8000/health
```

The response includes classifier status:

```json
{
  "status": "healthy",
  "components": {
    "retriever": "ready",
    "generator": "ready",
    "classifier": "ready"
  },
  "vector_store_count": 1234
}
```

## Programmatic Access

You can also use the classifier directly in Python:

```python
from src.classification.feedback_classifier import feedback_classifier

# Load model (if not already loaded)
feedback_classifier.load_model()

# Classify
result = feedback_classifier.classify(
    text="I can't log in!",
    language="en"
)

print(result)
# {
#   'text': "I can't log in!",
#   'language': 'en',
#   'level1': 'Technical',
#   'level1_confidence': 0.9672,
#   'level2': 'Login',
#   'level2_confidence': 0.8687
# }
```

## Troubleshooting

### Model Not Found
If you get "Model not found" errors:
1. Ensure you've trained the model using the notebook
2. Check that these files exist:
   - `notebooks/models/multitask/best_model/`
   - `notebooks/label_mappings_multitask.json`

### Out of Memory (OOM)
If training fails due to memory:
1. Close other applications
2. Reduce batch size in the notebook (already set to 2)
3. Use Google Colab with free GPU
4. Use a smaller model (distilbert-base-multilingual-cased)

### Low Confidence Scores
If predictions have low confidence:
1. Ensure the text is relevant to the categories
2. Check that the language parameter matches the text
3. Consider retraining with more data

