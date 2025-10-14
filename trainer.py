import torch
from tqdm.notebook import tqdm

# Entrenamiento y Evaluacion
def train_loop(model, device, train_loader, criterion, optimizer):
    """Función para una época de entrenamiento."""
    model.train() #Entrenamiento
    running_loss = 0.0
    correct_predictions = 0

    for inputs, labels in tqdm(train_loader, desc="Entrenando"):
        #mover los datoa a CPU o GPU
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad() #reinicio en 0 del los gadientes
        outputs = model(inputs) #hacer una prediccion (forward pass)
        loss = criterion(outputs, labels) #calculo de la perdida
        loss.backward() #calcular gradientes (back propagation)
        optimizer.step() #actualizar los pesos

        #Calcular metricas
        running_loss += loss.item() *inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct_predictions += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = correct_predictions.double() / len(train_loader.dataset)
    return epoch_loss, epoch_acc.item()


def val_loop(model, device, validation_loader,criterion):
    """Función para una época de validación."""
    model.eval() #Evaluacion
    running_loss = 0.0
    correct_predictions = 0

    with torch.no_grad(): 
        for inputs, labels in tqdm(validation_loader, desc="Validadndo"):
            #mover los datoa a CPU o GPU
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs) #hacer una prediccion (forward pass)
            loss = criterion(outputs, labels) #calculo de la perdida

            #Calcular metricas
            running_loss += loss.item() *inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(validation_loader.dataset)
    epoch_acc = correct_predictions.double() / len(validation_loader.dataset)
    return epoch_loss, epoch_acc.item()


