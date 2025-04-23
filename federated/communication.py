import os
import pickle
import numpy as np
import time
from copy import deepcopy


class Message:
    """
    Represents a message in the federated learning communication protocol.
    """

    def __init__(self, sender_id, receiver_id, message_type, payload=None):
        """
        Initialize a message.

        Args:
            sender_id: ID of the sender
            receiver_id: ID of the receiver ('server' for global server)
            message_type: Type of message ('model', 'data', 'update', etc.)
            payload: Message payload (model, data, etc.)
        """
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.message_type = message_type
        self.payload = payload
        self.timestamp = time.time()
        self.message_id = self._generate_message_id()

    def _generate_message_id(self):
        """Generate a unique message ID."""
        return f"{self.sender_id}_{self.receiver_id}_{self.message_type}_{int(self.timestamp)}"

    def serialize(self):
        """
        Serialize the message for transmission.

        Returns:
            Serialized message
        """
        return pickle.dumps(self)

    @staticmethod
    def deserialize(serialized_message):
        """
        Deserialize a message.

        Args:
            serialized_message: Serialized message

        Returns:
            Deserialized Message object
        """
        return pickle.loads(serialized_message)


class CommunicationManager:
    """
    Manages communication between federated learning clients and the server.
    """

    def __init__(self, simulation_mode=True, message_dir=None):
        """
        Initialize the communication manager.

        Args:
            simulation_mode: Whether to run in simulation mode (True) or real network mode (False)
            message_dir: Directory for storing messages in simulation mode
        """
        self.simulation_mode = simulation_mode

        if simulation_mode:
            self.message_dir = message_dir or 'messages'
            os.makedirs(self.message_dir, exist_ok=True)
        else:
            # In a real implementation, this would set up network connections
            raise NotImplementedError("Network mode not implemented yet")

        self.message_queue = {}
        self.received_messages = {}

    def send_message(self, message):
        """
        Send a message.

        Args:
            message: Message to send

        Returns:
            Success flag
        """
        if self.simulation_mode:
            return self._simulate_send_message(message)
        else:
            # In a real implementation, this would send over the network
            raise NotImplementedError("Network mode not implemented yet")

    def _simulate_send_message(self, message):
        """
        Simulate sending a message by storing it to a file.

        Args:
            message: Message to send

        Returns:
            Success flag
        """
        try:
            serialized_message = message.serialize()
            message_path = os.path.join(self.message_dir, message.message_id)

            with open(message_path, 'wb') as f:
                f.write(serialized_message)

            # Add to queue for the receiver
            receiver_id = message.receiver_id
            if receiver_id not in self.message_queue:
                self.message_queue[receiver_id] = []

            self.message_queue[receiver_id].append(message.message_id)

            return True
        except Exception as e:
            print(f"Error sending message: {str(e)}")
            return False

    def receive_messages(self, receiver_id):
        """
        Receive all pending messages for a receiver.

        Args:
            receiver_id: ID of the receiver

        Returns:
            List of received messages
        """
        if self.simulation_mode:
            return self._simulate_receive_messages(receiver_id)
        else:
            # In a real implementation, this would receive from the network
            raise NotImplementedError("Network mode not implemented yet")

    def _simulate_receive_messages(self, receiver_id):
        """
        Simulate receiving messages by loading from files.

        Args:
            receiver_id: ID of the receiver

        Returns:
            List of received messages
        """
        messages = []

        # Check message queue for this receiver
        if receiver_id in self.message_queue:
            message_ids = self.message_queue[receiver_id]

            for message_id in message_ids:
                message_path = os.path.join(self.message_dir, message_id)

                try:
                    if os.path.exists(message_path):
                        with open(message_path, 'rb') as f:
                            serialized_message = f.read()

                        message = Message.deserialize(serialized_message)
                        messages.append(message)

                        # Store in received messages
                        if receiver_id not in self.received_messages:
                            self.received_messages[receiver_id] = []

                        self.received_messages[receiver_id].append(message)

                        # Remove from queue
                        self.message_queue[receiver_id].remove(message_id)
                except Exception as e:
                    print(f"Error receiving message {message_id}: {str(e)}")

        return messages


class FederatedCommunicator:
    """
    Base class for federated learning communicators (clients and server).
    """

    def __init__(self, entity_id, comm_manager=None):
        """
        Initialize the federated communicator.

        Args:
            entity_id: ID of this entity
            comm_manager: Communication manager
        """
        self.entity_id = entity_id
        self.comm_manager = comm_manager or CommunicationManager()

    def send_model(self, receiver_id, model):
        """
        Send a model to another entity.

        Args:
            receiver_id: ID of the receiver
            model: Model to send

        Returns:
            Success flag
        """
        # Create a message with the model
        message = Message(
            sender_id=self.entity_id,
            receiver_id=receiver_id,
            message_type='model',
            payload=deepcopy(model)
        )

        # Send the message
        return self.comm_manager.send_message(message)

    def send_data(self, receiver_id, X, y):
        """
        Send data to another entity.

        Args:
            receiver_id: ID of the receiver
            X: Feature matrix
            y: Target vector

        Returns:
            Success flag
        """
        # Create a message with the data
        message = Message(
            sender_id=self.entity_id,
            receiver_id=receiver_id,
            message_type='data',
            payload=(X, y)
        )

        # Send the message
        return self.comm_manager.send_message(message)

    def send_update(self, receiver_id, update_type, payload):
        """
        Send an update message.

        Args:
            receiver_id: ID of the receiver
            update_type: Type of update ('weights', 'parameters', etc.)
            payload: Update payload

        Returns:
            Success flag
        """
        # Create a message with the update
        message = Message(
            sender_id=self.entity_id,
            receiver_id=receiver_id,
            message_type=f'update_{update_type}',
            payload=payload
        )

        # Send the message
        return self.comm_manager.send_message(message)

    def receive_messages(self):
        """
        Receive all pending messages.

        Returns:
            List of received messages
        """
        return self.comm_manager.receive_messages(self.entity_id)

    def process_messages(self):
        """
        Process all pending messages.

        Returns:
            Dictionary of processed messages by type
        """
        messages = self.receive_messages()
        processed = {'model': [], 'data': [], 'update': []}

        for message in messages:
            if message.message_type == 'model':
                processed['model'].append((message.sender_id, message.payload))
            elif message.message_type == 'data':
                processed['data'].append((message.sender_id, message.payload))
            elif message.message_type.startswith('update_'):
                processed['update'].append((message.sender_id, message.message_type, message.payload))

        return processed


class ClientCommunicator(FederatedCommunicator):
    """
    Communicator for federated learning clients.
    """

    def __init__(self, client_id, comm_manager=None):
        """
        Initialize the client communicator.

        Args:
            client_id: ID of this client
            comm_manager: Communication manager
        """
        super().__init__(client_id, comm_manager)
        self.server_id = 'server'

    def send_model_to_server(self, model):
        """
        Send a model to the server.

        Args:
            model: Model to send

        Returns:
            Success flag
        """
        return self.send_model(self.server_id, model)

    def send_data_to_server(self, X, y):
        """
        Send data to the server.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            Success flag
        """
        return self.send_data(self.server_id, X, y)

    def receive_global_model(self):
        """
        Receive the global model from the server.

        Returns:
            Global model or None if not received
        """
        processed = self.process_messages()

        for sender_id, model in processed['model']:
            if sender_id == self.server_id:
                return model

        return None


class ServerCommunicator(FederatedCommunicator):
    """
    Communicator for the federated learning server.
    """

    def __init__(self, comm_manager=None):
        """
        Initialize the server communicator.

        Args:
            comm_manager: Communication manager
        """
        super().__init__('server', comm_manager)
        self.clients = set()

    def add_client(self, client_id):
        """
        Add a client to the server.

        Args:
            client_id: ID of the client to add
        """
        self.clients.add(client_id)

    def broadcast_model(self, model):
        """
        Broadcast a model to all clients.

        Args:
            model: Model to broadcast

        Returns:
            Dictionary mapping client IDs to success flags
        """
        results = {}

        for client_id in self.clients:
            results[client_id] = self.send_model(client_id, model)

        return results

    def collect_client_models(self):
        """
        Collect models from all clients.

        Returns:
            Dictionary mapping client IDs to their models
        """
        processed = self.process_messages()
        client_models = {}

        for sender_id, model in processed['model']:
            client_models[sender_id] = model

        return client_models

    def collect_client_data(self):
        """
        Collect data from all clients.

        Returns:
            Dictionary mapping client IDs to their data (X, y)
        """
        processed = self.process_messages()
        client_data = {}

        for sender_id, data in processed['data']:
            client_data[sender_id] = data

        return client_data