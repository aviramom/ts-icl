
# Implements the dataset of UCR


import os
import torch
import pandas as pd
from torch.utils.data import Dataset, ConcatDataset
from scipy.io import arff

# ── Load domain descriptions and label descriptions from ucr_descriptions/ ──
_UCR_DESC_DIR = os.path.join(os.path.dirname(__file__), '..', 'ucr_descriptions')

def _load_ucr_txt_descriptions():
    descriptions = {}
    if not os.path.isdir(_UCR_DESC_DIR):
        return descriptions, {}
    for dataset_name in os.listdir(_UCR_DESC_DIR):
        txt_path = os.path.join(_UCR_DESC_DIR, dataset_name, 'description.txt')
        if not os.path.isfile(txt_path):
            continue
        with open(txt_path, encoding='utf-8') as f:
            text = f.read()
        if text.strip():
            descriptions[dataset_name] = text
    return descriptions, {}

UCR_DESCRIPTIONS, UCR_LABEL_DESCRIPTIONS = _load_ucr_txt_descriptions()

class UCRDataset(Dataset):
    def __init__(self, ucr_path, split: str = "train"):
        """
        Load UCR dataset from ARFF or TSV format.

        Args:
            ucr_path: Path to dataset file or directory containing TRAIN/TEST files
            split: 'train' or 'test'
        """
        dataset_name = os.path.basename(ucr_path)
        train_path = os.path.join(ucr_path, f"{dataset_name}_TRAIN.arff")
        test_path = os.path.join(ucr_path, f"{dataset_name}_TEST.arff")

        if split == "train":
            df = self._load_file(train_path)
        elif split == "test":
            df = self._load_file(test_path)
        else:
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")

        self.desc = self.UCR_DESCRIPTIONS.get(dataset_name)

        labels_raw = df.iloc[:, -1].astype('int64')
        self.labels = torch.tensor(labels_raw, dtype=torch.long)
        self.data = torch.tensor(df.iloc[:, :-1].values.astype('float32'), dtype=torch.float32)

    def _load_file(self, file_path):
        if file_path.endswith('.arff'):
            return self._load_arff(file_path)
        return pd.read_csv(file_path, header=None, sep='\t')

    def _load_arff(self, arff_path):
        data, meta = arff.loadarff(arff_path)
        return pd.DataFrame(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    

    UCR_DESCRIPTIONS = {
        # --- IMAGE / SHAPE DATASETS ---
        "Adiac": "Image data representing the outlines of diatoms (unicellular algae). The task is to identify the species based on the outline shape converted to a time series.",
        "ArrowHead": "Image data representing the outlines of arrowheads. The task is to classify the type of projectile point.",
        "BeetleFly": "Image data representing the outlines of beetles and flies. The task is to distinguish between the two insect types.",
        "BirdChicken": "Image data representing the outlines of birds and chickens. The task is to distinguish between the two based on shape.",
        "DiatomSizeReduction": "Image data representing diatom outlines. The task is to classify the reduction size class of the diatom.",
        "DistalPhalanxOutlineAgeGroup": "Image data of X-rays of hands. The task is to classify the age group (0-6 years, 7-12 years, 13-19 years) based on the outline of the distal phalanx.",
        "DistalPhalanxOutlineCorrect": "Image data of X-rays of hands. The task is to determine if the segmentation of the distal phalanx outline is correct or incorrect.",
        "DistalPhalanxTW": "Image data of X-rays of hands. The task is to classify the bone age using Time Warping distances.",
        "FaceAll": "Image data representing the outlines of human faces. The task is to distinguish between different individuals.",
        "FaceFour": "Image data representing the outlines of human faces. The task is to distinguish between four specific individuals.",
        "FacesUCR": "Image data representing the outlines of human faces from the UCR graduate student body.",
        "FiftyWords": "Image data representing the handwritten outlines of 50 common words.",
        "Fish": "Image data representing the outlines of various fish species.",
        "HandOutlines": "Image data representing the outlines of human hands.",
        "Herring": "Image data representing the outlines of herring fish.",
        "InsectWingbeatSound": "Audio data converted to frequency domain representing the flying sounds of insects. The task is to classify the insect species.",
        "MiddlePhalanxOutlineAgeGroup": "Image data of X-rays of hands. The task is to classify the age group based on the outline of the middle phalanx.",
        "MiddlePhalanxOutlineCorrect": "Image data of X-rays of hands. The task is to determine if the segmentation of the middle phalanx outline is correct.",
        "MiddlePhalanxTW": "Image data of X-rays of hands (Middle Phalanx) using Time Warping distances.",
        "OSULeaf": "Image data representing the outlines of leaves from the OSU dataset.",
        "ProximalPhalanxOutlineAgeGroup": "Image data of X-rays of hands. The task is to classify the age group based on the outline of the proximal phalanx.",
        "ProximalPhalanxOutlineCorrect": "Image data of X-rays of hands. The task is to determine if the segmentation of the proximal phalanx outline is correct.",
        "ProximalPhalanxTW": "Image data of X-rays of hands (Proximal Phalanx) using Time Warping distances.",
        "ShapesAll": "Image data representing the outlines of various binary shapes.",
        "SwedishLeaf": "Image data representing the outlines of leaves from Swedish tree species.",
        "Symbols": "Image data representing the outlines of drawn symbols.",
        "MedicalImages": "Image data derived from medical imaging histograms.",
        "WordSynonyms": "Image data representing the handwritten outlines of words.",
        "Yoga": "Image data representing the outlines of an actor performing yoga poses. The task is to identify the gender of the actor.",

        # --- SENSOR / DEVICE DATASETS ---
        "BME": "Sensor data from a BME280 sensor measuring temperature, humidity, and pressure.",
        "Car": "Sensor data from passing cars. The task is to classify the type of vehicle.",
        "Chinatown": "Traffic flow data (pedestrian counts) in Melbourne's Chinatown.",
        "ChlorineConcentration": "Simulated data modeling chlorine concentration levels in a water distribution system.",
        "Computers": "Electricity consumption data from desktop computers in different usage states.",
        "Crop": "Satellite image time series used to classify different crop types.",
        "DodgerLoopDay": "Traffic sensor data from the Dodger Stadium loop. The task is to classify the day of the week.",
        "DodgerLoopGame": "Traffic sensor data. The task is to determine if there is a baseball game on that day.",
        "DodgerLoopWeekend": "Traffic sensor data. The task is to determine if it is a weekend or weekday.",
        "Earthquakes": "Seismic data. The task is to predict earthquake events.",
        "ElectricDevices": "Electricity consumption profiles of various household devices.",
        "FordA": "Automotive sensor data measuring engine noise. The task is to diagnose whether a specific symptom exists in the subsystem.",
        "FordB": "Automotive sensor data measuring engine noise. The task is to diagnose whether a specific symptom exists (similar to FordA but with different training data).",
        "FreezerRegularTrain": "Sensor data from a freezer. The task is to classify the freezer's operating state.",
        "FreezerSmallTrain": "Sensor data from a freezer (small training set).",
        "HouseTwenty": "Electricity usage monitoring data from a smart home.",
        "Lightning2": "Sensor data recording lightning power density. The task is to classify the lightning strike type.",
        "Lightning7": "Sensor data recording lightning power density (7 classes).",
        "MelbournePedestrian": "Pedestrian count data from sensors in Melbourne.",
        "Plane": "Sensor data from sensors mounted on a plane.",
        "PowerCons": "Individual household electric power consumption data.",
        "Rock": "Sensor data classifying rock types.",
        "SemgHandGenderCh2": "EMG (Electromyography) sensor data from hand muscles. The task is to classify gender.",
        "SemgHandMovementCh2": "EMG (Electromyography) sensor data. The task is to classify hand movements.",
        "SemgHandSubjectCh2": "EMG (Electromyography) sensor data. The task is to identify the subject.",
        "SmoothSubspace": "Synthetic sensor data.",
        "SonyAIBORobotSurface1": "Accelerometer data from a Sony AIBO robot. The task is to identify the surface it is walking on.",
        "SonyAIBORobotSurface2": "Accelerometer data from a Sony AIBO robot (Dataset 2).",
        "Trace": "Instrumentation data simulating a transient classification task in a nuclear power plant.",
        "UWaveGestureLibraryAll": "Accelerometer data tracking hand gestures.",
        "UWaveGestureLibraryX": "Accelerometer data tracking hand gestures (X-axis).",
        "UWaveGestureLibraryY": "Accelerometer data tracking hand gestures (Y-axis).",
        "UWaveGestureLibraryZ": "Accelerometer data tracking hand gestures (Z-axis).",
        "Wafer": "Sensor data from semiconductor wafer processing. The task is to detect abnormal processing.",

        # --- MOTION / HAR (Human Activity Recognition) ---
        "CricketX": "Accelerometer data (X-axis) of an umpire's hand signals in the game of cricket.",
        "CricketY": "Accelerometer data (Y-axis) of an umpire's hand signals in the game of cricket.",
        "CricketZ": "Accelerometer data (Z-axis) of an umpire's hand signals in the game of cricket.",
        "GunPoint": "Motion tracking data of an actor's hand. The task is to classify whether the actor is drawing a gun from a hip holster or simply pointing a finger.",
        "GunPointAgeSpan": "Motion tracking data distinguishing between young and old actors performing the GunPoint action.",
        "GunPointMaleVersusFemale": "Motion tracking data distinguishing between male and female actors performing the GunPoint action.",
        "GunPointOldVersusYoung": "Motion tracking data distinguishing between old and young actors.",
        "Haptics": "Data from a haptic device tracing a predefined path.",
        "InlineSkate": "Motion data from professional inline speed skating. The task is to classify the technique used.",
        "PickupGestureWiimoteZ": "Accelerometer data from a Nintendo Wiimote. The task is to recognize the 'pickup' gesture.",
        "ShakeGestureWiimoteZ": "Accelerometer data from a Nintendo Wiimote. The task is to recognize the 'shake' gesture.",
        "ToeSegmentation1": "Motion capture data of the toe during walking. The task is to identify the start/end of a step.",
        "ToeSegmentation2": "Motion capture data of the toe (Dataset 2).",
        "Worms": "Motion data tracking the movement of C. elegans worms. The task is to classify the genotype.",
        "WormsTwoClass": "Motion data tracking the movement of C. elegans worms (Binary classification).",

        # --- ECG / EEG / MEDICAL ---
        "CinCECGTorso": "ECG data recorded from the torso. The task is to classify distinct people.",
        "ECG200": "Electrical activity recorded from the human heart (ECG). The task is to detect cardiac abnormalities (Myocardial Infarction).",
        "ECG5000": "ECG data extracted from a long Holter recording. The task is to classify heartbeat types.",
        "ECGFiveDays": "ECG data recorded from a single subject on two different days.",
        "NonInvasiveFetalECGThorax1": "Fetal ECG data recorded from the mother's thorax.",
        "NonInvasiveFetalECGThorax2": "Fetal ECG data (Dataset 2).",
        "TwoLeadECG": "ECG data from two leads.",
        "EOGHorizontalSignal": "Electro-oculography (EOG) data measuring eye movement (Horizontal).",
        "EOGVerticalSignal": "Electro-oculography (EOG) data measuring eye movement (Vertical).",
        
        # --- SPECTROGRAPHS / CHEMOMETRICS ---
        "Beef": "Spectrographic analysis of beef. The task is to classify whether the beef is pure or adulterated with offal.",
        "Coffee": "Spectrographic data of coffee samples. The task is to distinguish between Robusta and Arabica coffee beans.",
        "EthanolLevel": "Spectrographic analysis of spirits to determine ethanol concentration.",
        "Ham": "Spectrographic analysis of ham. The task is to classify the type of ham.",
        "Meat": "Spectrographic analysis of chicken, pork, and turkey.",
        "OliveOil": "Spectrographic analysis of olive oils. The task is to determine the geographic origin of the oil.",
        "PigAirwayPressure": "Medical sensor data measuring airway pressure in pigs.",
        "PigArtPressure": "Medical sensor data measuring arterial pressure in pigs.",
        "PigCVP": "Medical sensor data measuring central venous pressure in pigs.",
        "Strawberry": "Spectrographic analysis of strawberries. The task is to classify the variety.",
        "Wine": "Spectrographic analysis of wine. The task is to classify the grape variety.",

        # --- SIMULATED / SYNTHETIC ---
        "CBF": "Synthetic 'Cylinder-Bell-Funnel' data. The task is to classify the time series into one of these three generated shape patterns.",
        "Mallat": "Simulated data based on Mallat's scattering transform examples.",
        "SyntheticControl": "Synthetic control chart patterns (e.g., Normal, Cyclic, Increasing trend).",
        "TwoPatterns": "Synthetic data combining two specific patterns.",
        "MoteStrain": "Sensor data from a mote node measuring resistor strain.",
    }

