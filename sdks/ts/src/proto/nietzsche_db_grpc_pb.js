// GENERATED CODE -- DO NOT EDIT!

'use strict';
var grpc = require('@grpc/grpc-js');
var nietzsche_db_pb = require('./nietzsche_db_pb.js');

function serialize_nietzsche_BatchInsertRequest(arg) {
  if (!(arg instanceof nietzsche_db_pb.BatchInsertRequest)) {
    throw new Error('Expected argument of type nietzsche.BatchInsertRequest');
  }
  return Buffer.from(arg.serializeBinary());
}

function deserialize_nietzsche_BatchInsertRequest(buffer_arg) {
  return nietzsche_db_pb.BatchInsertRequest.deserializeBinary(new Uint8Array(buffer_arg));
}

function serialize_nietzsche_BatchSearchRequest(arg) {
  if (!(arg instanceof nietzsche_db_pb.BatchSearchRequest)) {
    throw new Error('Expected argument of type nietzsche.BatchSearchRequest');
  }
  return Buffer.from(arg.serializeBinary());
}

function deserialize_nietzsche_BatchSearchRequest(buffer_arg) {
  return nietzsche_db_pb.BatchSearchRequest.deserializeBinary(new Uint8Array(buffer_arg));
}

function serialize_nietzsche_BatchSearchResponse(arg) {
  if (!(arg instanceof nietzsche_db_pb.BatchSearchResponse)) {
    throw new Error('Expected argument of type nietzsche.BatchSearchResponse');
  }
  return Buffer.from(arg.serializeBinary());
}

function deserialize_nietzsche_BatchSearchResponse(buffer_arg) {
  return nietzsche_db_pb.BatchSearchResponse.deserializeBinary(new Uint8Array(buffer_arg));
}

function serialize_nietzsche_CollectionStatsRequest(arg) {
  if (!(arg instanceof nietzsche_db_pb.CollectionStatsRequest)) {
    throw new Error('Expected argument of type nietzsche.CollectionStatsRequest');
  }
  return Buffer.from(arg.serializeBinary());
}

function deserialize_nietzsche_CollectionStatsRequest(buffer_arg) {
  return nietzsche_db_pb.CollectionStatsRequest.deserializeBinary(new Uint8Array(buffer_arg));
}

function serialize_nietzsche_CollectionStatsResponse(arg) {
  if (!(arg instanceof nietzsche_db_pb.CollectionStatsResponse)) {
    throw new Error('Expected argument of type nietzsche.CollectionStatsResponse');
  }
  return Buffer.from(arg.serializeBinary());
}

function deserialize_nietzsche_CollectionStatsResponse(buffer_arg) {
  return nietzsche_db_pb.CollectionStatsResponse.deserializeBinary(new Uint8Array(buffer_arg));
}

function serialize_nietzsche_ConfigUpdate(arg) {
  if (!(arg instanceof nietzsche_db_pb.ConfigUpdate)) {
    throw new Error('Expected argument of type nietzsche.ConfigUpdate');
  }
  return Buffer.from(arg.serializeBinary());
}

function deserialize_nietzsche_ConfigUpdate(buffer_arg) {
  return nietzsche_db_pb.ConfigUpdate.deserializeBinary(new Uint8Array(buffer_arg));
}

function serialize_nietzsche_CreateCollectionRequest(arg) {
  if (!(arg instanceof nietzsche_db_pb.CreateCollectionRequest)) {
    throw new Error('Expected argument of type nietzsche.CreateCollectionRequest');
  }
  return Buffer.from(arg.serializeBinary());
}

function deserialize_nietzsche_CreateCollectionRequest(buffer_arg) {
  return nietzsche_db_pb.CreateCollectionRequest.deserializeBinary(new Uint8Array(buffer_arg));
}

function serialize_nietzsche_DeleteCollectionRequest(arg) {
  if (!(arg instanceof nietzsche_db_pb.DeleteCollectionRequest)) {
    throw new Error('Expected argument of type nietzsche.DeleteCollectionRequest');
  }
  return Buffer.from(arg.serializeBinary());
}

function deserialize_nietzsche_DeleteCollectionRequest(buffer_arg) {
  return nietzsche_db_pb.DeleteCollectionRequest.deserializeBinary(new Uint8Array(buffer_arg));
}

function serialize_nietzsche_DeleteRequest(arg) {
  if (!(arg instanceof nietzsche_db_pb.DeleteRequest)) {
    throw new Error('Expected argument of type nietzsche.DeleteRequest');
  }
  return Buffer.from(arg.serializeBinary());
}

function deserialize_nietzsche_DeleteRequest(buffer_arg) {
  return nietzsche_db_pb.DeleteRequest.deserializeBinary(new Uint8Array(buffer_arg));
}

function serialize_nietzsche_DeleteResponse(arg) {
  if (!(arg instanceof nietzsche_db_pb.DeleteResponse)) {
    throw new Error('Expected argument of type nietzsche.DeleteResponse');
  }
  return Buffer.from(arg.serializeBinary());
}

function deserialize_nietzsche_DeleteResponse(buffer_arg) {
  return nietzsche_db_pb.DeleteResponse.deserializeBinary(new Uint8Array(buffer_arg));
}

function serialize_nietzsche_DigestRequest(arg) {
  if (!(arg instanceof nietzsche_db_pb.DigestRequest)) {
    throw new Error('Expected argument of type nietzsche.DigestRequest');
  }
  return Buffer.from(arg.serializeBinary());
}

function deserialize_nietzsche_DigestRequest(buffer_arg) {
  return nietzsche_db_pb.DigestRequest.deserializeBinary(new Uint8Array(buffer_arg));
}

function serialize_nietzsche_DigestResponse(arg) {
  if (!(arg instanceof nietzsche_db_pb.DigestResponse)) {
    throw new Error('Expected argument of type nietzsche.DigestResponse');
  }
  return Buffer.from(arg.serializeBinary());
}

function deserialize_nietzsche_DigestResponse(buffer_arg) {
  return nietzsche_db_pb.DigestResponse.deserializeBinary(new Uint8Array(buffer_arg));
}

function serialize_nietzsche_Empty(arg) {
  if (!(arg instanceof nietzsche_db_pb.Empty)) {
    throw new Error('Expected argument of type nietzsche.Empty');
  }
  return Buffer.from(arg.serializeBinary());
}

function deserialize_nietzsche_Empty(buffer_arg) {
  return nietzsche_db_pb.Empty.deserializeBinary(new Uint8Array(buffer_arg));
}

function serialize_nietzsche_InsertRequest(arg) {
  if (!(arg instanceof nietzsche_db_pb.InsertRequest)) {
    throw new Error('Expected argument of type nietzsche.InsertRequest');
  }
  return Buffer.from(arg.serializeBinary());
}

function deserialize_nietzsche_InsertRequest(buffer_arg) {
  return nietzsche_db_pb.InsertRequest.deserializeBinary(new Uint8Array(buffer_arg));
}

function serialize_nietzsche_InsertResponse(arg) {
  if (!(arg instanceof nietzsche_db_pb.InsertResponse)) {
    throw new Error('Expected argument of type nietzsche.InsertResponse');
  }
  return Buffer.from(arg.serializeBinary());
}

function deserialize_nietzsche_InsertResponse(buffer_arg) {
  return nietzsche_db_pb.InsertResponse.deserializeBinary(new Uint8Array(buffer_arg));
}

function serialize_nietzsche_InsertTextRequest(arg) {
  if (!(arg instanceof nietzsche_db_pb.InsertTextRequest)) {
    throw new Error('Expected argument of type nietzsche.InsertTextRequest');
  }
  return Buffer.from(arg.serializeBinary());
}

function deserialize_nietzsche_InsertTextRequest(buffer_arg) {
  return nietzsche_db_pb.InsertTextRequest.deserializeBinary(new Uint8Array(buffer_arg));
}

function serialize_nietzsche_ListCollectionsResponse(arg) {
  if (!(arg instanceof nietzsche_db_pb.ListCollectionsResponse)) {
    throw new Error('Expected argument of type nietzsche.ListCollectionsResponse');
  }
  return Buffer.from(arg.serializeBinary());
}

function deserialize_nietzsche_ListCollectionsResponse(buffer_arg) {
  return nietzsche_db_pb.ListCollectionsResponse.deserializeBinary(new Uint8Array(buffer_arg));
}

function serialize_nietzsche_MonitorRequest(arg) {
  if (!(arg instanceof nietzsche_db_pb.MonitorRequest)) {
    throw new Error('Expected argument of type nietzsche.MonitorRequest');
  }
  return Buffer.from(arg.serializeBinary());
}

function deserialize_nietzsche_MonitorRequest(buffer_arg) {
  return nietzsche_db_pb.MonitorRequest.deserializeBinary(new Uint8Array(buffer_arg));
}

function serialize_nietzsche_RebuildIndexRequest(arg) {
  if (!(arg instanceof nietzsche_db_pb.RebuildIndexRequest)) {
    throw new Error('Expected argument of type nietzsche.RebuildIndexRequest');
  }
  return Buffer.from(arg.serializeBinary());
}

function deserialize_nietzsche_RebuildIndexRequest(buffer_arg) {
  return nietzsche_db_pb.RebuildIndexRequest.deserializeBinary(new Uint8Array(buffer_arg));
}

function serialize_nietzsche_ReplicationLog(arg) {
  if (!(arg instanceof nietzsche_db_pb.ReplicationLog)) {
    throw new Error('Expected argument of type nietzsche.ReplicationLog');
  }
  return Buffer.from(arg.serializeBinary());
}

function deserialize_nietzsche_ReplicationLog(buffer_arg) {
  return nietzsche_db_pb.ReplicationLog.deserializeBinary(new Uint8Array(buffer_arg));
}

function serialize_nietzsche_ReplicationRequest(arg) {
  if (!(arg instanceof nietzsche_db_pb.ReplicationRequest)) {
    throw new Error('Expected argument of type nietzsche.ReplicationRequest');
  }
  return Buffer.from(arg.serializeBinary());
}

function deserialize_nietzsche_ReplicationRequest(buffer_arg) {
  return nietzsche_db_pb.ReplicationRequest.deserializeBinary(new Uint8Array(buffer_arg));
}

function serialize_nietzsche_SearchRequest(arg) {
  if (!(arg instanceof nietzsche_db_pb.SearchRequest)) {
    throw new Error('Expected argument of type nietzsche.SearchRequest');
  }
  return Buffer.from(arg.serializeBinary());
}

function deserialize_nietzsche_SearchRequest(buffer_arg) {
  return nietzsche_db_pb.SearchRequest.deserializeBinary(new Uint8Array(buffer_arg));
}

function serialize_nietzsche_SearchResponse(arg) {
  if (!(arg instanceof nietzsche_db_pb.SearchResponse)) {
    throw new Error('Expected argument of type nietzsche.SearchResponse');
  }
  return Buffer.from(arg.serializeBinary());
}

function deserialize_nietzsche_SearchResponse(buffer_arg) {
  return nietzsche_db_pb.SearchResponse.deserializeBinary(new Uint8Array(buffer_arg));
}

function serialize_nietzsche_StatusResponse(arg) {
  if (!(arg instanceof nietzsche_db_pb.StatusResponse)) {
    throw new Error('Expected argument of type nietzsche.StatusResponse');
  }
  return Buffer.from(arg.serializeBinary());
}

function deserialize_nietzsche_StatusResponse(buffer_arg) {
  return nietzsche_db_pb.StatusResponse.deserializeBinary(new Uint8Array(buffer_arg));
}

function serialize_nietzsche_SystemStats(arg) {
  if (!(arg instanceof nietzsche_db_pb.SystemStats)) {
    throw new Error('Expected argument of type nietzsche.SystemStats');
  }
  return Buffer.from(arg.serializeBinary());
}

function deserialize_nietzsche_SystemStats(buffer_arg) {
  return nietzsche_db_pb.SystemStats.deserializeBinary(new Uint8Array(buffer_arg));
}


var DatabaseService = exports.DatabaseService = {
  // Collection Management
createCollection: {
    path: '/nietzsche.Database/CreateCollection',
    requestStream: false,
    responseStream: false,
    requestType: nietzsche_db_pb.CreateCollectionRequest,
    responseType: nietzsche_db_pb.StatusResponse,
    requestSerialize: serialize_nietzsche_CreateCollectionRequest,
    requestDeserialize: deserialize_nietzsche_CreateCollectionRequest,
    responseSerialize: serialize_nietzsche_StatusResponse,
    responseDeserialize: deserialize_nietzsche_StatusResponse,
  },
  deleteCollection: {
    path: '/nietzsche.Database/DeleteCollection',
    requestStream: false,
    responseStream: false,
    requestType: nietzsche_db_pb.DeleteCollectionRequest,
    responseType: nietzsche_db_pb.StatusResponse,
    requestSerialize: serialize_nietzsche_DeleteCollectionRequest,
    requestDeserialize: deserialize_nietzsche_DeleteCollectionRequest,
    responseSerialize: serialize_nietzsche_StatusResponse,
    responseDeserialize: deserialize_nietzsche_StatusResponse,
  },
  listCollections: {
    path: '/nietzsche.Database/ListCollections',
    requestStream: false,
    responseStream: false,
    requestType: nietzsche_db_pb.Empty,
    responseType: nietzsche_db_pb.ListCollectionsResponse,
    requestSerialize: serialize_nietzsche_Empty,
    requestDeserialize: deserialize_nietzsche_Empty,
    responseSerialize: serialize_nietzsche_ListCollectionsResponse,
    responseDeserialize: deserialize_nietzsche_ListCollectionsResponse,
  },
  getCollectionStats: {
    path: '/nietzsche.Database/GetCollectionStats',
    requestStream: false,
    responseStream: false,
    requestType: nietzsche_db_pb.CollectionStatsRequest,
    responseType: nietzsche_db_pb.CollectionStatsResponse,
    requestSerialize: serialize_nietzsche_CollectionStatsRequest,
    requestDeserialize: deserialize_nietzsche_CollectionStatsRequest,
    responseSerialize: serialize_nietzsche_CollectionStatsResponse,
    responseDeserialize: deserialize_nietzsche_CollectionStatsResponse,
  },
  // Insert vectors
insert: {
    path: '/nietzsche.Database/Insert',
    requestStream: false,
    responseStream: false,
    requestType: nietzsche_db_pb.InsertRequest,
    responseType: nietzsche_db_pb.InsertResponse,
    requestSerialize: serialize_nietzsche_InsertRequest,
    requestDeserialize: deserialize_nietzsche_InsertRequest,
    responseSerialize: serialize_nietzsche_InsertResponse,
    responseDeserialize: deserialize_nietzsche_InsertResponse,
  },
  batchInsert: {
    path: '/nietzsche.Database/BatchInsert',
    requestStream: false,
    responseStream: false,
    requestType: nietzsche_db_pb.BatchInsertRequest,
    responseType: nietzsche_db_pb.InsertResponse,
    requestSerialize: serialize_nietzsche_BatchInsertRequest,
    requestDeserialize: deserialize_nietzsche_BatchInsertRequest,
    responseSerialize: serialize_nietzsche_InsertResponse,
    responseDeserialize: deserialize_nietzsche_InsertResponse,
  },
  insertText: {
    path: '/nietzsche.Database/InsertText',
    requestStream: false,
    responseStream: false,
    requestType: nietzsche_db_pb.InsertTextRequest,
    responseType: nietzsche_db_pb.InsertResponse,
    requestSerialize: serialize_nietzsche_InsertTextRequest,
    requestDeserialize: deserialize_nietzsche_InsertTextRequest,
    responseSerialize: serialize_nietzsche_InsertResponse,
    responseDeserialize: deserialize_nietzsche_InsertResponse,
  },
  // Delete vectors
delete: {
    path: '/nietzsche.Database/Delete',
    requestStream: false,
    responseStream: false,
    requestType: nietzsche_db_pb.DeleteRequest,
    responseType: nietzsche_db_pb.DeleteResponse,
    requestSerialize: serialize_nietzsche_DeleteRequest,
    requestDeserialize: deserialize_nietzsche_DeleteRequest,
    responseSerialize: serialize_nietzsche_DeleteResponse,
    responseDeserialize: deserialize_nietzsche_DeleteResponse,
  },
  // Search (ANN)
search: {
    path: '/nietzsche.Database/Search',
    requestStream: false,
    responseStream: false,
    requestType: nietzsche_db_pb.SearchRequest,
    responseType: nietzsche_db_pb.SearchResponse,
    requestSerialize: serialize_nietzsche_SearchRequest,
    requestDeserialize: deserialize_nietzsche_SearchRequest,
    responseSerialize: serialize_nietzsche_SearchResponse,
    responseDeserialize: deserialize_nietzsche_SearchResponse,
  },
  // Batch Search (ANN)
searchBatch: {
    path: '/nietzsche.Database/SearchBatch',
    requestStream: false,
    responseStream: false,
    requestType: nietzsche_db_pb.BatchSearchRequest,
    responseType: nietzsche_db_pb.BatchSearchResponse,
    requestSerialize: serialize_nietzsche_BatchSearchRequest,
    requestDeserialize: deserialize_nietzsche_BatchSearchRequest,
    responseSerialize: serialize_nietzsche_BatchSearchResponse,
    responseDeserialize: deserialize_nietzsche_BatchSearchResponse,
  },
  // Stream statistics for TUI (Global or Collection tailored)
monitor: {
    path: '/nietzsche.Database/Monitor',
    requestStream: false,
    responseStream: true,
    requestType: nietzsche_db_pb.MonitorRequest,
    responseType: nietzsche_db_pb.SystemStats,
    requestSerialize: serialize_nietzsche_MonitorRequest,
    requestDeserialize: deserialize_nietzsche_MonitorRequest,
    responseSerialize: serialize_nietzsche_SystemStats,
    responseDeserialize: deserialize_nietzsche_SystemStats,
  },
  // Admin Controls
triggerSnapshot: {
    path: '/nietzsche.Database/TriggerSnapshot',
    requestStream: false,
    responseStream: false,
    requestType: nietzsche_db_pb.Empty,
    responseType: nietzsche_db_pb.StatusResponse,
    requestSerialize: serialize_nietzsche_Empty,
    requestDeserialize: deserialize_nietzsche_Empty,
    responseSerialize: serialize_nietzsche_StatusResponse,
    responseDeserialize: deserialize_nietzsche_StatusResponse,
  },
  triggerVacuum: {
    path: '/nietzsche.Database/TriggerVacuum',
    requestStream: false,
    responseStream: false,
    requestType: nietzsche_db_pb.Empty,
    responseType: nietzsche_db_pb.StatusResponse,
    requestSerialize: serialize_nietzsche_Empty,
    requestDeserialize: deserialize_nietzsche_Empty,
    responseSerialize: serialize_nietzsche_StatusResponse,
    responseDeserialize: deserialize_nietzsche_StatusResponse,
  },
  // Dynamic Configuration
configure: {
    path: '/nietzsche.Database/Configure',
    requestStream: false,
    responseStream: false,
    requestType: nietzsche_db_pb.ConfigUpdate,
    responseType: nietzsche_db_pb.StatusResponse,
    requestSerialize: serialize_nietzsche_ConfigUpdate,
    requestDeserialize: deserialize_nietzsche_ConfigUpdate,
    responseSerialize: serialize_nietzsche_StatusResponse,
    responseDeserialize: deserialize_nietzsche_StatusResponse,
  },
  // Replication (Leader -> Follower)
replicate: {
    path: '/nietzsche.Database/Replicate',
    requestStream: false,
    responseStream: true,
    requestType: nietzsche_db_pb.ReplicationRequest,
    responseType: nietzsche_db_pb.ReplicationLog,
    requestSerialize: serialize_nietzsche_ReplicationRequest,
    requestDeserialize: deserialize_nietzsche_ReplicationRequest,
    responseSerialize: serialize_nietzsche_ReplicationLog,
    responseDeserialize: deserialize_nietzsche_ReplicationLog,
  },
  getDigest: {
    path: '/nietzsche.Database/GetDigest',
    requestStream: false,
    responseStream: false,
    requestType: nietzsche_db_pb.DigestRequest,
    responseType: nietzsche_db_pb.DigestResponse,
    requestSerialize: serialize_nietzsche_DigestRequest,
    requestDeserialize: deserialize_nietzsche_DigestRequest,
    responseSerialize: serialize_nietzsche_DigestResponse,
    responseDeserialize: deserialize_nietzsche_DigestResponse,
  },
  rebuildIndex: {
    path: '/nietzsche.Database/RebuildIndex',
    requestStream: false,
    responseStream: false,
    requestType: nietzsche_db_pb.RebuildIndexRequest,
    responseType: nietzsche_db_pb.StatusResponse,
    requestSerialize: serialize_nietzsche_RebuildIndexRequest,
    requestDeserialize: deserialize_nietzsche_RebuildIndexRequest,
    responseSerialize: serialize_nietzsche_StatusResponse,
    responseDeserialize: deserialize_nietzsche_StatusResponse,
  },
};

exports.DatabaseClient = grpc.makeGenericClientConstructor(DatabaseService, 'Database');
