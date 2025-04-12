from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from PIL import Image, UnidentifiedImageError
from transformers import ViTFeatureExtractor, ViTForImageClassification
import torch
import io

app = FastAPI()

# Load model and feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained('wambugu71/crop_leaf_diseases_vit')
model = ViTForImageClassification.from_pretrained(
    'wambugu1738/crop_leaf_diseases_vit',
    ignore_mismatched_sizes=True
)

# Remedies dictionary
remedies = {
    "Corn___Common_Rust": {
        "en": {
            "disease": "Corn - Common Rust",
            "remedy": "Apply fungicides like Mancozeb or Azoxystrobin at early signs.",
            "medicine": "Mancozeb 75% WP or Azoxystrobin 23% SC"
        },
        "ta": {
            "disease": "மக்காச்சோளம் - பொது காளான் புண்",
            "remedy": "முதன்மை அறிகுறிகளில் Mancozeb அல்லது Azoxystrobin போன்ற பூஞ்சைக் கொல்லிகளை தெளிக்கவும்.",
            "medicine": "Mancozeb 75% WP அல்லது Azoxystrobin 23% SC"
        }
    },
    "Corn___Gray_Leaf_Spot": {
        "en": {
            "disease": "Corn - Gray Leaf Spot",
            "remedy": "Use fungicides like Trifloxystrobin or Propiconazole.",
            "medicine": "Trifloxystrobin 25% WG or Propiconazole 25% EC"
        },
        "ta": {
            "disease": "மக்காச்சோளம் - சாம்பல் இலை புள்ளி",
            "remedy": "Trifloxystrobin அல்லது Propiconazole போன்ற பூஞ்சைக் கொல்லிகளை பயன்படுத்தவும்.",
            "medicine": "Trifloxystrobin 25% WG அல்லது Propiconazole 25% EC"
        }
    },
    "Corn___Healthy": {
        "en": {
            "disease": "Corn - Healthy",
            "remedy": "No disease detected. Maintain crop with proper watering and fertilization.",
            "medicine": "N/A"
        },
        "ta": {
            "disease": "மக்காச்சோளம் - நலமாக உள்ளது",
            "remedy": "எந்த நோயும் கண்டறியப்படவில்லை. சரியான நீர்ப்பாசனம் மற்றும் உரமிடலை தொடரவும்.",
            "medicine": "தொடர்ந்து பராமரிக்கவும்"
        }
    },
    "Wheat___Brown_Rust": {
        "en": {
            "disease": "Wheat - Brown Rust",
            "remedy": "Use fungicides like Propiconazole or Tebuconazole.",
            "medicine": "Propiconazole 25% EC or Tebuconazole 25% EC"
        },
        "ta": {
            "disease": "கோதுமை - பழுப்பு காளான் புண்",
            "remedy": "Propiconazole அல்லது Tebuconazole போன்ற பூஞ்சைக் கொல்லிகளை பயன்படுத்தவும்.",
            "medicine": "Propiconazole 25% EC அல்லது Tebuconazole 25% EC"
        }
    },
    "Wheat___Yellow_Rust": {
        "en": {
            "disease": "Wheat - Yellow Rust",
            "remedy": "Apply fungicides such as Mancozeb or Hexaconazole.",
            "medicine": "Mancozeb 75% WP or Hexaconazole 5% EC"
        },
        "ta": {
            "disease": "கோதுமை - மஞ்சள் காளான் புண்",
            "remedy": "Mancozeb அல்லது Hexaconazole போன்ற பூஞ்சைக் கொல்லிகளை தெளிக்கவும்.",
            "medicine": "Mancozeb 75% WP அல்லது Hexaconazole 5% EC"
        }
    },
    "Wheat___Healthy": {
        "en": {
            "disease": "Wheat - Healthy",
            "remedy": "No disease found. Continue with routine care and nutrient management.",
            "medicine": "N/A"
        },
        "ta": {
            "disease": "கோதுமை - நலமாக உள்ளது",
            "remedy": "நோயில்லை. வழக்கமான பராமரிப்பு மற்றும் சத்துணவுகளை தொடரவும்.",
            "medicine": "தொடர்ந்து பராமரிக்கவும்"
        }
    },
    "Potato___Early_Blight": {
        "en": {
            "disease": "Potato - Early Blight",
            "remedy": "Use fungicides like Chlorothalonil or Mancozeb.",
            "medicine": "Chlorothalonil 75% WP or Mancozeb 75% WP"
        },
        "ta": {
            "disease": "உருளைக்கிழங்கு - ஆரம்ப பிளைட்",
            "remedy": "Chlorothalonil அல்லது Mancozeb போன்ற பூஞ்சைக் கொல்லிகளை பயன்படுத்தவும்.",
            "medicine": "Chlorothalonil 75% WP அல்லது Mancozeb 75% WP"
        }
    },
    "Potato___Late_Blight": {
        "en": {
            "disease": "Potato - Late Blight",
            "remedy": "Apply fungicides like Metalaxyl or Dimethomorph.",
            "medicine": "Metalaxyl 8% + Mancozeb 64% WP or Dimethomorph 50% WP"
        },
        "ta": {
            "disease": "உருளைக்கிழங்கு - முந்தைய பிளைட்",
            "remedy": "Metalaxyl அல்லது Dimethomorph போன்ற பூஞ்சைக் கொல்லிகளை தெளிக்கவும்.",
            "medicine": "Metalaxyl 8% + Mancozeb 64% WP அல்லது Dimethomorph 50% WP"
        }
    },
    "Potato___Healthy": {
        "en": {
            "disease": "Potato - Healthy",
            "remedy": "No disease detected. Regular crop maintenance is advised.",
            "medicine": "N/A"
        },
        "ta": {
            "disease": "உருளைக்கிழங்கு - நலமாக உள்ளது",
            "remedy": "நோயில்லை. வழக்கமான பராமரிப்பை தொடரவும்.",
            "medicine": "தொடர்ந்து பராமரிக்கவும்"
        }
    },
    "Rice___Brown_Spot": {
        "en": {
            "disease": "Rice - Brown Spot",
            "remedy": "Use fungicides like Carbendazim or Tricyclazole.",
            "medicine": "Carbendazim 50% WP or Tricyclazole 75% WP"
        },
        "ta": {
            "disease": "அரிசி - பழுப்பு புள்ளி",
            "remedy": "Carbendazim அல்லது Tricyclazole போன்ற பூஞ்சைக் கொல்லிகளை தெளிக்கவும்.",
            "medicine": "Carbendazim 50% WP அல்லது Tricyclazole 75% WP"
        }
    },
    "Rice___Leaf_Blast": {
        "en": {
            "disease": "Rice - Leaf Blast",
            "remedy": "Apply fungicides such as Tricyclazole or Isoprothiolane.",
            "medicine": "Tricyclazole 75% WP or Isoprothiolane 40% EC"
        },
        "ta": {
            "disease": "அரிசி - இலை வெடிப்பு",
            "remedy": "Tricyclazole அல்லது Isoprothiolane போன்ற பூஞ்சைக் கொல்லிகளை தெளிக்கவும்.",
            "medicine": "Tricyclazole 75% WP அல்லது Isoprothiolane 40% EC"
        }
    },
    "Rice___Healthy": {
        "en": {
            "disease": "Rice - Healthy",
            "remedy": "No disease detected. Ensure regular irrigation and balanced nutrition.",
            "medicine": "N/A"
        },
        "ta": {
            "disease": "அரிசி - நலமாக உள்ளது",
            "remedy": "நோயில்லை. சரியான நீர்ப்பாசனம் மற்றும் சத்துணவுகளை தொடரவும்.",
            "medicine": "தொடர்ந்து பராமரிக்கவும்"
        }
    },
    "Invalid": {
        "en": {
            "disease": "Invalid - No crop disease detected",
            "remedy": "Please upload a clear image of a crop leaf.",
            "medicine": "N/A"
        },
        "ta": {
            "disease": "தவறான படம் - நோய்கள் இல்லை",
            "remedy": "தயவுசெய்து தெளிவான பயிர் இலை படத்தை பதிவேற்றவும்.",
            "medicine": "பொருந்தாது"
        }
    }
}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...), lang: str = Query("en")):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image format")

    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        predicted_label = model.config.id2label[predicted_class_idx]

    remedy_info = remedies.get(predicted_label, {}).get(lang, {
        "disease": predicted_label,
        "remedy": "No remedy info available.",
        "medicine": "N/A"
    })

    return {
        "predicted_class": predicted_label,
        "disease_name": remedy_info["disease"],
        "remedy": remedy_info["remedy"],
        "medicine": remedy_info["medicine"]
    }
