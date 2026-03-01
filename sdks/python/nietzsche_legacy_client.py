import grpc
import nietzsche_db_pb2
import nietzsche_db_pb2_grpc
import numpy as np

class NietzscheBaseClient:
    def __init__(self, host="localhost:50051"):
        self.channel = grpc.insecure_channel(host)
        self.stub = nietzsche_db_pb2_grpc.DatabaseStub(self.channel)

    def search(self, vector: np.ndarray, top_k: int = 10):
        # Validate Poincare norm < 1
        if np.linalg.norm(vector) >= 1:
            raise ValueError("Vector must be inside Poincaré ball")
            
        req = nietzsche_db_pb2.SearchRequest(
            vector=vector.tolist(), 
            top_k=top_k
        )
        return self.stub.Search(req)
