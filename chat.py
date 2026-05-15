from datetime import date

from src.agent import MODEL, run_agent

today = date.today()
default_quartal = f"{(today.month - 1) // 3 + 1}/{today.year}"

print(f"GOPilot — model: {MODEL}  (empty dictation to quit)\n")
patient_id = input("Patient-ID [P001]: ").strip() or "P001"
quartal = input(f"Quartal [{default_quartal}]: ").strip() or default_quartal

while True:
    user_input = input("\nDiktat: ").strip()
    if not user_input:
        break

    result = run_agent(user_input, patient_id=patient_id, quartal=quartal)
    print(f"\nGOPilot: {result['response']}\n")
