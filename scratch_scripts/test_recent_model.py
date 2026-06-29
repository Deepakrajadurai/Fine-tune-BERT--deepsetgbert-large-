import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def run_test():
    model_path = "models/legal_model"
    if not os.path.exists(model_path):
        print(f"Error: Model path {model_path} does not exist!")
        return

    print(f"Loading model and tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Model loaded on {device}.")

    # Test cases with proper German umlauts
    texts = {
        "1. Human Sample (App default - proper umlauts)": (
            "Die Bundesregierung hat heute im Kabinett die neue Digitalstrategie beschlossen. "
            "Bundesminister Karl Lauterbach erklärte bei der anschließenden Pressekonferenz, dass die "
            "Investitionen in die digitale Infrastruktur in den kommenden Jahren erheblich steigen werden. "
            "\"Wir müssen aufholen, was in den letzten Jahrzehnten versäumt wurde\", sagte er vor versammelten Journalisten."
        ),
        "2. AI Sample (App default - generic Digitalisierung - proper umlauts)": (
            "Die Digitalisierung in Deutschland schreitet voran und bietet zahlreiche Möglichkeiten "
            "für Wirtschaft und Gesellschaft. Durch gezielte Investitionen in moderne Infrastruktur und "
            "innovative Technologien kann Deutschland seine Wettbewerbsfähigkeit auf dem globalen Markt "
            "nachhaltig stärken. Es ist wichtig, dass alle Beteiligten gemeinsam an diesem Zukunftsprojekt "
            "arbeiten, um die Potenziale der digitalen Transformation vollständig auszuschöpfen."
        ),
        "3. Simulated AI Bundestag Debate Speech (proper umlauts)": (
            "Meine Damen und Herren, liebe Kolleginnen und Kollegen, heute debattieren wir über den "
            "Gesetzentwurf der Bundesregierung zur Digitalisierung des Justizwesens. Als Mitglied des "
            "Bundestages möchte ich betonen, dass diese Reform von entscheidender Bedeutung ist. Die "
            "Einführung der elektronischen Akte wird die Effizienz unserer Gerichte erheblich steigern. "
            "Dennoch müssen wir sicherstellen, dass der Datenschutz gewahrt bleibt."
        ),
        "4. Real German Legal Statute / Paragraph (AI-generated style - proper umlauts)": (
            "Paragraph 1 Anwendungsbereich und Zweck des Gesetzes. (1) Dieses Gesetz regelt die Bedingungen "
            "und das Verfahren für die Gewährung von Bundesfinanzhilfen zur Förderung der Digitalisierung "
            "der Verwaltung in den Ländern und Kommunen. Ziel ist es, die Effizienz und Bürgerfreundlichkeit "
            "der öffentlichen Verwaltung nachhaltig zu steigern. (2) Der Bund stellt zu diesem Zweck Finanzmittel "
            "im Rahmen des vereinbarten Sondervermögens zur Verfügung. Die Verteilung der Mittel erfolgt nach "
            "einem festgelegten Schlüssel, der die Einwohnerzahl und die spezifischen Bedarfe der Länder "
            "berücksichtigt. (3) Die Empfänger der Finanzhilfen sind verpflichtet, dem Bundesministerium "
            "für Digitales jährlich über die Verwendung der Mittel und den Fortschritt der geförderten "
            "Projekte Bericht zu erstatten."
        )
    }

    for label, text in texts.items():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).squeeze().tolist()
            predicted_class_id = torch.argmax(logits, dim=-1).item()
            predicted_label = "AI" if predicted_class_id == 1 else "Human"

        print("\n" + "="*60)
        print(f"Label: {label}")
        print(f"Text snippet: {text[:100]}...")
        print(f"Logits: {logits.squeeze().tolist()}")
        print(f"Probabilities - Human: {probs[0]:.4f}, AI: {probs[1]:.4f}")
        print(f"Verdict: {predicted_label}")

if __name__ == "__main__":
    run_test()
