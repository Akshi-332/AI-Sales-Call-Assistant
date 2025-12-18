from transformers import pipeline

def analyze_text(text):
    model = pipeline("sentiment-analysis")
    result = model(text[:512])[0]
    return {"label": result["label"], "score": round(result["score"], 2)}

if __name__ == "__main__":
    text = "The offer is very good and affordable."
    print(analyze_text(text))
