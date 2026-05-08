import torch

print(f"Czy ROCm jest dostępny: {torch.cuda.is_available()}")
print(f"Liczba urządzeń: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Nazwa urządzenia: {torch.cuda.get_device_name(0)}")