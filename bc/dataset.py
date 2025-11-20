import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.io


class BCDataset(Dataset):
    """
    Dataset for Behavior Cloning
    Loads data from .mpz files
    """
    def __init__(self, data_path, transform=None):
        """
        Args:
            data_path: Path to .mpz file
            transform: Optional transform to be applied on a sample
        """
        self.data_path = data_path
        self.transform = transform
        
        # Load data from .mpz file
        self.data = self._load_mpz_data(data_path)
        
        # Extract inputs and outputs
        # TODO: 실제 데이터 구조에 맞게 수정 필요
        self.inputs = self.data['inputs']  # Shape: (N, 72)
        self.outputs = self.data['outputs']  # Shape: (N, 12)
        
        # TODO: 전처리 (Preprocessing) 구현
        # 
        # 전처리 예시:
        # 
        # 1. 정규화 (Normalization)
        #    self.inputs = (self.inputs - self.inputs.mean(axis=0)) / (self.inputs.std(axis=0) + 1e-8)
        #    self.outputs = (self.outputs - self.outputs.mean(axis=0)) / (self.outputs.std(axis=0) + 1e-8)
        #
        # 2. Min-Max 스케일링
        #    self.inputs = (self.inputs - self.inputs.min(axis=0)) / (self.inputs.max(axis=0) - self.inputs.min(axis=0) + 1e-8)
        #
        # 3. 특징 엔지니어링
        #    - 추가 특징 생성
        #    - 특징 선택
        #
        # 4. 데이터 증강 (Data Augmentation)
        #    - 노이즈 추가
        #    - 회전, 변환 등
        
        # Convert to tensors
        self.inputs = torch.FloatTensor(self.inputs)
        self.outputs = torch.FloatTensor(self.outputs)
        
        print(f"Loaded dataset: {len(self.inputs)} samples")
        print(f"Input shape: {self.inputs.shape}")
        print(f"Output shape: {self.outputs.shape}")
    
    def _load_mpz_data(self, file_path):
        """
        Load data from .mpz file
        
        Args:
            file_path: Path to .mpz file
        
        Returns:
            Dictionary containing loaded data with keys:
            - 'inputs': numpy array of shape (N, 72)
            - 'outputs': numpy array of shape (N, 12)
        """
        # TODO: .mpz 파일 로드 및 전처리 구현
        # 
        # .mpz 파일 로드 방법 예시:
        # 
        # 방법 1: numpy.loadz 사용
        # data = np.load(file_path, allow_pickle=True)
        # inputs = data['inputs']  # 또는 data.item()['inputs']
        # outputs = data['outputs']
        #
        # 방법 2: scipy.io.loadmat 사용 (MATLAB .mat 파일인 경우)
        # data = scipy.io.loadmat(file_path)
        # inputs = data['inputs']
        # outputs = data['outputs']
        #
        # 전처리 예시:
        # - Normalization: inputs = (inputs - mean) / std
        # - Feature scaling: inputs = (inputs - min) / (max - min)
        # - Feature selection/extraction
        #
        # 반환 형식:
        # return {
        #     'inputs': inputs,  # Shape: (N, 72)
        #     'outputs': outputs  # Shape: (N, 12)
        # }
        
        raise NotImplementedError(
            "TODO: Implement .mpz file loading and preprocessing.\n"
            "Please implement the _load_mpz_data method to load your .mpz file\n"
            "and return a dictionary with 'inputs' (N, 72) and 'outputs' (N, 12) keys."
        )
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset
        
        Args:
            idx: Index of the sample
        
        Returns:
            Tuple of (input, output)
        """
        sample_input = self.inputs[idx]
        sample_output = self.outputs[idx]
        
        if self.transform:
            sample_input = self.transform(sample_input)
            sample_output = self.transform(sample_output)
        
        return sample_input, sample_output

