// package: nietzsche
// file: nietzsche_db.proto

/* tslint:disable */
/* eslint-disable */

import * as grpc from "@grpc/grpc-js";
import * as nietzsche_db_pb from "./nietzsche_db_pb";

interface IDatabaseService extends grpc.ServiceDefinition<grpc.UntypedServiceImplementation> {
    createCollection: IDatabaseService_ICreateCollection;
    deleteCollection: IDatabaseService_IDeleteCollection;
    listCollections: IDatabaseService_IListCollections;
    getCollectionStats: IDatabaseService_IGetCollectionStats;
    insert: IDatabaseService_IInsert;
    batchInsert: IDatabaseService_IBatchInsert;
    insertText: IDatabaseService_IInsertText;
    delete: IDatabaseService_IDelete;
    search: IDatabaseService_ISearch;
    searchBatch: IDatabaseService_ISearchBatch;
    monitor: IDatabaseService_IMonitor;
    triggerSnapshot: IDatabaseService_ITriggerSnapshot;
    triggerVacuum: IDatabaseService_ITriggerVacuum;
    configure: IDatabaseService_IConfigure;
    replicate: IDatabaseService_IReplicate;
    getDigest: IDatabaseService_IGetDigest;
    rebuildIndex: IDatabaseService_IRebuildIndex;
}

interface IDatabaseService_ICreateCollection extends grpc.MethodDefinition<nietzsche_db_pb.CreateCollectionRequest, nietzsche_db_pb.StatusResponse> {
    path: "/nietzsche.Database/CreateCollection";
    requestStream: false;
    responseStream: false;
    requestSerialize: grpc.serialize<nietzsche_db_pb.CreateCollectionRequest>;
    requestDeserialize: grpc.deserialize<nietzsche_db_pb.CreateCollectionRequest>;
    responseSerialize: grpc.serialize<nietzsche_db_pb.StatusResponse>;
    responseDeserialize: grpc.deserialize<nietzsche_db_pb.StatusResponse>;
}
interface IDatabaseService_IDeleteCollection extends grpc.MethodDefinition<nietzsche_db_pb.DeleteCollectionRequest, nietzsche_db_pb.StatusResponse> {
    path: "/nietzsche.Database/DeleteCollection";
    requestStream: false;
    responseStream: false;
    requestSerialize: grpc.serialize<nietzsche_db_pb.DeleteCollectionRequest>;
    requestDeserialize: grpc.deserialize<nietzsche_db_pb.DeleteCollectionRequest>;
    responseSerialize: grpc.serialize<nietzsche_db_pb.StatusResponse>;
    responseDeserialize: grpc.deserialize<nietzsche_db_pb.StatusResponse>;
}
interface IDatabaseService_IListCollections extends grpc.MethodDefinition<nietzsche_db_pb.Empty, nietzsche_db_pb.ListCollectionsResponse> {
    path: "/nietzsche.Database/ListCollections";
    requestStream: false;
    responseStream: false;
    requestSerialize: grpc.serialize<nietzsche_db_pb.Empty>;
    requestDeserialize: grpc.deserialize<nietzsche_db_pb.Empty>;
    responseSerialize: grpc.serialize<nietzsche_db_pb.ListCollectionsResponse>;
    responseDeserialize: grpc.deserialize<nietzsche_db_pb.ListCollectionsResponse>;
}
interface IDatabaseService_IGetCollectionStats extends grpc.MethodDefinition<nietzsche_db_pb.CollectionStatsRequest, nietzsche_db_pb.CollectionStatsResponse> {
    path: "/nietzsche.Database/GetCollectionStats";
    requestStream: false;
    responseStream: false;
    requestSerialize: grpc.serialize<nietzsche_db_pb.CollectionStatsRequest>;
    requestDeserialize: grpc.deserialize<nietzsche_db_pb.CollectionStatsRequest>;
    responseSerialize: grpc.serialize<nietzsche_db_pb.CollectionStatsResponse>;
    responseDeserialize: grpc.deserialize<nietzsche_db_pb.CollectionStatsResponse>;
}
interface IDatabaseService_IInsert extends grpc.MethodDefinition<nietzsche_db_pb.InsertRequest, nietzsche_db_pb.InsertResponse> {
    path: "/nietzsche.Database/Insert";
    requestStream: false;
    responseStream: false;
    requestSerialize: grpc.serialize<nietzsche_db_pb.InsertRequest>;
    requestDeserialize: grpc.deserialize<nietzsche_db_pb.InsertRequest>;
    responseSerialize: grpc.serialize<nietzsche_db_pb.InsertResponse>;
    responseDeserialize: grpc.deserialize<nietzsche_db_pb.InsertResponse>;
}
interface IDatabaseService_IBatchInsert extends grpc.MethodDefinition<nietzsche_db_pb.BatchInsertRequest, nietzsche_db_pb.InsertResponse> {
    path: "/nietzsche.Database/BatchInsert";
    requestStream: false;
    responseStream: false;
    requestSerialize: grpc.serialize<nietzsche_db_pb.BatchInsertRequest>;
    requestDeserialize: grpc.deserialize<nietzsche_db_pb.BatchInsertRequest>;
    responseSerialize: grpc.serialize<nietzsche_db_pb.InsertResponse>;
    responseDeserialize: grpc.deserialize<nietzsche_db_pb.InsertResponse>;
}
interface IDatabaseService_IInsertText extends grpc.MethodDefinition<nietzsche_db_pb.InsertTextRequest, nietzsche_db_pb.InsertResponse> {
    path: "/nietzsche.Database/InsertText";
    requestStream: false;
    responseStream: false;
    requestSerialize: grpc.serialize<nietzsche_db_pb.InsertTextRequest>;
    requestDeserialize: grpc.deserialize<nietzsche_db_pb.InsertTextRequest>;
    responseSerialize: grpc.serialize<nietzsche_db_pb.InsertResponse>;
    responseDeserialize: grpc.deserialize<nietzsche_db_pb.InsertResponse>;
}
interface IDatabaseService_IDelete extends grpc.MethodDefinition<nietzsche_db_pb.DeleteRequest, nietzsche_db_pb.DeleteResponse> {
    path: "/nietzsche.Database/Delete";
    requestStream: false;
    responseStream: false;
    requestSerialize: grpc.serialize<nietzsche_db_pb.DeleteRequest>;
    requestDeserialize: grpc.deserialize<nietzsche_db_pb.DeleteRequest>;
    responseSerialize: grpc.serialize<nietzsche_db_pb.DeleteResponse>;
    responseDeserialize: grpc.deserialize<nietzsche_db_pb.DeleteResponse>;
}
interface IDatabaseService_ISearch extends grpc.MethodDefinition<nietzsche_db_pb.SearchRequest, nietzsche_db_pb.SearchResponse> {
    path: "/nietzsche.Database/Search";
    requestStream: false;
    responseStream: false;
    requestSerialize: grpc.serialize<nietzsche_db_pb.SearchRequest>;
    requestDeserialize: grpc.deserialize<nietzsche_db_pb.SearchRequest>;
    responseSerialize: grpc.serialize<nietzsche_db_pb.SearchResponse>;
    responseDeserialize: grpc.deserialize<nietzsche_db_pb.SearchResponse>;
}
interface IDatabaseService_ISearchBatch extends grpc.MethodDefinition<nietzsche_db_pb.BatchSearchRequest, nietzsche_db_pb.BatchSearchResponse> {
    path: "/nietzsche.Database/SearchBatch";
    requestStream: false;
    responseStream: false;
    requestSerialize: grpc.serialize<nietzsche_db_pb.BatchSearchRequest>;
    requestDeserialize: grpc.deserialize<nietzsche_db_pb.BatchSearchRequest>;
    responseSerialize: grpc.serialize<nietzsche_db_pb.BatchSearchResponse>;
    responseDeserialize: grpc.deserialize<nietzsche_db_pb.BatchSearchResponse>;
}
interface IDatabaseService_IMonitor extends grpc.MethodDefinition<nietzsche_db_pb.MonitorRequest, nietzsche_db_pb.SystemStats> {
    path: "/nietzsche.Database/Monitor";
    requestStream: false;
    responseStream: true;
    requestSerialize: grpc.serialize<nietzsche_db_pb.MonitorRequest>;
    requestDeserialize: grpc.deserialize<nietzsche_db_pb.MonitorRequest>;
    responseSerialize: grpc.serialize<nietzsche_db_pb.SystemStats>;
    responseDeserialize: grpc.deserialize<nietzsche_db_pb.SystemStats>;
}
interface IDatabaseService_ITriggerSnapshot extends grpc.MethodDefinition<nietzsche_db_pb.Empty, nietzsche_db_pb.StatusResponse> {
    path: "/nietzsche.Database/TriggerSnapshot";
    requestStream: false;
    responseStream: false;
    requestSerialize: grpc.serialize<nietzsche_db_pb.Empty>;
    requestDeserialize: grpc.deserialize<nietzsche_db_pb.Empty>;
    responseSerialize: grpc.serialize<nietzsche_db_pb.StatusResponse>;
    responseDeserialize: grpc.deserialize<nietzsche_db_pb.StatusResponse>;
}
interface IDatabaseService_ITriggerVacuum extends grpc.MethodDefinition<nietzsche_db_pb.Empty, nietzsche_db_pb.StatusResponse> {
    path: "/nietzsche.Database/TriggerVacuum";
    requestStream: false;
    responseStream: false;
    requestSerialize: grpc.serialize<nietzsche_db_pb.Empty>;
    requestDeserialize: grpc.deserialize<nietzsche_db_pb.Empty>;
    responseSerialize: grpc.serialize<nietzsche_db_pb.StatusResponse>;
    responseDeserialize: grpc.deserialize<nietzsche_db_pb.StatusResponse>;
}
interface IDatabaseService_IConfigure extends grpc.MethodDefinition<nietzsche_db_pb.ConfigUpdate, nietzsche_db_pb.StatusResponse> {
    path: "/nietzsche.Database/Configure";
    requestStream: false;
    responseStream: false;
    requestSerialize: grpc.serialize<nietzsche_db_pb.ConfigUpdate>;
    requestDeserialize: grpc.deserialize<nietzsche_db_pb.ConfigUpdate>;
    responseSerialize: grpc.serialize<nietzsche_db_pb.StatusResponse>;
    responseDeserialize: grpc.deserialize<nietzsche_db_pb.StatusResponse>;
}
interface IDatabaseService_IReplicate extends grpc.MethodDefinition<nietzsche_db_pb.ReplicationRequest, nietzsche_db_pb.ReplicationLog> {
    path: "/nietzsche.Database/Replicate";
    requestStream: false;
    responseStream: true;
    requestSerialize: grpc.serialize<nietzsche_db_pb.ReplicationRequest>;
    requestDeserialize: grpc.deserialize<nietzsche_db_pb.ReplicationRequest>;
    responseSerialize: grpc.serialize<nietzsche_db_pb.ReplicationLog>;
    responseDeserialize: grpc.deserialize<nietzsche_db_pb.ReplicationLog>;
}
interface IDatabaseService_IGetDigest extends grpc.MethodDefinition<nietzsche_db_pb.DigestRequest, nietzsche_db_pb.DigestResponse> {
    path: "/nietzsche.Database/GetDigest";
    requestStream: false;
    responseStream: false;
    requestSerialize: grpc.serialize<nietzsche_db_pb.DigestRequest>;
    requestDeserialize: grpc.deserialize<nietzsche_db_pb.DigestRequest>;
    responseSerialize: grpc.serialize<nietzsche_db_pb.DigestResponse>;
    responseDeserialize: grpc.deserialize<nietzsche_db_pb.DigestResponse>;
}
interface IDatabaseService_IRebuildIndex extends grpc.MethodDefinition<nietzsche_db_pb.RebuildIndexRequest, nietzsche_db_pb.StatusResponse> {
    path: "/nietzsche.Database/RebuildIndex";
    requestStream: false;
    responseStream: false;
    requestSerialize: grpc.serialize<nietzsche_db_pb.RebuildIndexRequest>;
    requestDeserialize: grpc.deserialize<nietzsche_db_pb.RebuildIndexRequest>;
    responseSerialize: grpc.serialize<nietzsche_db_pb.StatusResponse>;
    responseDeserialize: grpc.deserialize<nietzsche_db_pb.StatusResponse>;
}

export const DatabaseService: IDatabaseService;

export interface IDatabaseServer extends grpc.UntypedServiceImplementation {
    createCollection: grpc.handleUnaryCall<nietzsche_db_pb.CreateCollectionRequest, nietzsche_db_pb.StatusResponse>;
    deleteCollection: grpc.handleUnaryCall<nietzsche_db_pb.DeleteCollectionRequest, nietzsche_db_pb.StatusResponse>;
    listCollections: grpc.handleUnaryCall<nietzsche_db_pb.Empty, nietzsche_db_pb.ListCollectionsResponse>;
    getCollectionStats: grpc.handleUnaryCall<nietzsche_db_pb.CollectionStatsRequest, nietzsche_db_pb.CollectionStatsResponse>;
    insert: grpc.handleUnaryCall<nietzsche_db_pb.InsertRequest, nietzsche_db_pb.InsertResponse>;
    batchInsert: grpc.handleUnaryCall<nietzsche_db_pb.BatchInsertRequest, nietzsche_db_pb.InsertResponse>;
    insertText: grpc.handleUnaryCall<nietzsche_db_pb.InsertTextRequest, nietzsche_db_pb.InsertResponse>;
    delete: grpc.handleUnaryCall<nietzsche_db_pb.DeleteRequest, nietzsche_db_pb.DeleteResponse>;
    search: grpc.handleUnaryCall<nietzsche_db_pb.SearchRequest, nietzsche_db_pb.SearchResponse>;
    searchBatch: grpc.handleUnaryCall<nietzsche_db_pb.BatchSearchRequest, nietzsche_db_pb.BatchSearchResponse>;
    monitor: grpc.handleServerStreamingCall<nietzsche_db_pb.MonitorRequest, nietzsche_db_pb.SystemStats>;
    triggerSnapshot: grpc.handleUnaryCall<nietzsche_db_pb.Empty, nietzsche_db_pb.StatusResponse>;
    triggerVacuum: grpc.handleUnaryCall<nietzsche_db_pb.Empty, nietzsche_db_pb.StatusResponse>;
    configure: grpc.handleUnaryCall<nietzsche_db_pb.ConfigUpdate, nietzsche_db_pb.StatusResponse>;
    replicate: grpc.handleServerStreamingCall<nietzsche_db_pb.ReplicationRequest, nietzsche_db_pb.ReplicationLog>;
    getDigest: grpc.handleUnaryCall<nietzsche_db_pb.DigestRequest, nietzsche_db_pb.DigestResponse>;
    rebuildIndex: grpc.handleUnaryCall<nietzsche_db_pb.RebuildIndexRequest, nietzsche_db_pb.StatusResponse>;
}

export interface IDatabaseClient {
    createCollection(request: nietzsche_db_pb.CreateCollectionRequest, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.StatusResponse) => void): grpc.ClientUnaryCall;
    createCollection(request: nietzsche_db_pb.CreateCollectionRequest, metadata: grpc.Metadata, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.StatusResponse) => void): grpc.ClientUnaryCall;
    createCollection(request: nietzsche_db_pb.CreateCollectionRequest, metadata: grpc.Metadata, options: Partial<grpc.CallOptions>, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.StatusResponse) => void): grpc.ClientUnaryCall;
    deleteCollection(request: nietzsche_db_pb.DeleteCollectionRequest, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.StatusResponse) => void): grpc.ClientUnaryCall;
    deleteCollection(request: nietzsche_db_pb.DeleteCollectionRequest, metadata: grpc.Metadata, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.StatusResponse) => void): grpc.ClientUnaryCall;
    deleteCollection(request: nietzsche_db_pb.DeleteCollectionRequest, metadata: grpc.Metadata, options: Partial<grpc.CallOptions>, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.StatusResponse) => void): grpc.ClientUnaryCall;
    listCollections(request: nietzsche_db_pb.Empty, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.ListCollectionsResponse) => void): grpc.ClientUnaryCall;
    listCollections(request: nietzsche_db_pb.Empty, metadata: grpc.Metadata, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.ListCollectionsResponse) => void): grpc.ClientUnaryCall;
    listCollections(request: nietzsche_db_pb.Empty, metadata: grpc.Metadata, options: Partial<grpc.CallOptions>, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.ListCollectionsResponse) => void): grpc.ClientUnaryCall;
    getCollectionStats(request: nietzsche_db_pb.CollectionStatsRequest, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.CollectionStatsResponse) => void): grpc.ClientUnaryCall;
    getCollectionStats(request: nietzsche_db_pb.CollectionStatsRequest, metadata: grpc.Metadata, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.CollectionStatsResponse) => void): grpc.ClientUnaryCall;
    getCollectionStats(request: nietzsche_db_pb.CollectionStatsRequest, metadata: grpc.Metadata, options: Partial<grpc.CallOptions>, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.CollectionStatsResponse) => void): grpc.ClientUnaryCall;
    insert(request: nietzsche_db_pb.InsertRequest, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.InsertResponse) => void): grpc.ClientUnaryCall;
    insert(request: nietzsche_db_pb.InsertRequest, metadata: grpc.Metadata, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.InsertResponse) => void): grpc.ClientUnaryCall;
    insert(request: nietzsche_db_pb.InsertRequest, metadata: grpc.Metadata, options: Partial<grpc.CallOptions>, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.InsertResponse) => void): grpc.ClientUnaryCall;
    batchInsert(request: nietzsche_db_pb.BatchInsertRequest, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.InsertResponse) => void): grpc.ClientUnaryCall;
    batchInsert(request: nietzsche_db_pb.BatchInsertRequest, metadata: grpc.Metadata, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.InsertResponse) => void): grpc.ClientUnaryCall;
    batchInsert(request: nietzsche_db_pb.BatchInsertRequest, metadata: grpc.Metadata, options: Partial<grpc.CallOptions>, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.InsertResponse) => void): grpc.ClientUnaryCall;
    insertText(request: nietzsche_db_pb.InsertTextRequest, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.InsertResponse) => void): grpc.ClientUnaryCall;
    insertText(request: nietzsche_db_pb.InsertTextRequest, metadata: grpc.Metadata, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.InsertResponse) => void): grpc.ClientUnaryCall;
    insertText(request: nietzsche_db_pb.InsertTextRequest, metadata: grpc.Metadata, options: Partial<grpc.CallOptions>, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.InsertResponse) => void): grpc.ClientUnaryCall;
    delete(request: nietzsche_db_pb.DeleteRequest, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.DeleteResponse) => void): grpc.ClientUnaryCall;
    delete(request: nietzsche_db_pb.DeleteRequest, metadata: grpc.Metadata, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.DeleteResponse) => void): grpc.ClientUnaryCall;
    delete(request: nietzsche_db_pb.DeleteRequest, metadata: grpc.Metadata, options: Partial<grpc.CallOptions>, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.DeleteResponse) => void): grpc.ClientUnaryCall;
    search(request: nietzsche_db_pb.SearchRequest, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.SearchResponse) => void): grpc.ClientUnaryCall;
    search(request: nietzsche_db_pb.SearchRequest, metadata: grpc.Metadata, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.SearchResponse) => void): grpc.ClientUnaryCall;
    search(request: nietzsche_db_pb.SearchRequest, metadata: grpc.Metadata, options: Partial<grpc.CallOptions>, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.SearchResponse) => void): grpc.ClientUnaryCall;
    searchBatch(request: nietzsche_db_pb.BatchSearchRequest, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.BatchSearchResponse) => void): grpc.ClientUnaryCall;
    searchBatch(request: nietzsche_db_pb.BatchSearchRequest, metadata: grpc.Metadata, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.BatchSearchResponse) => void): grpc.ClientUnaryCall;
    searchBatch(request: nietzsche_db_pb.BatchSearchRequest, metadata: grpc.Metadata, options: Partial<grpc.CallOptions>, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.BatchSearchResponse) => void): grpc.ClientUnaryCall;
    monitor(request: nietzsche_db_pb.MonitorRequest, options?: Partial<grpc.CallOptions>): grpc.ClientReadableStream<nietzsche_db_pb.SystemStats>;
    monitor(request: nietzsche_db_pb.MonitorRequest, metadata?: grpc.Metadata, options?: Partial<grpc.CallOptions>): grpc.ClientReadableStream<nietzsche_db_pb.SystemStats>;
    triggerSnapshot(request: nietzsche_db_pb.Empty, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.StatusResponse) => void): grpc.ClientUnaryCall;
    triggerSnapshot(request: nietzsche_db_pb.Empty, metadata: grpc.Metadata, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.StatusResponse) => void): grpc.ClientUnaryCall;
    triggerSnapshot(request: nietzsche_db_pb.Empty, metadata: grpc.Metadata, options: Partial<grpc.CallOptions>, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.StatusResponse) => void): grpc.ClientUnaryCall;
    triggerVacuum(request: nietzsche_db_pb.Empty, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.StatusResponse) => void): grpc.ClientUnaryCall;
    triggerVacuum(request: nietzsche_db_pb.Empty, metadata: grpc.Metadata, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.StatusResponse) => void): grpc.ClientUnaryCall;
    triggerVacuum(request: nietzsche_db_pb.Empty, metadata: grpc.Metadata, options: Partial<grpc.CallOptions>, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.StatusResponse) => void): grpc.ClientUnaryCall;
    configure(request: nietzsche_db_pb.ConfigUpdate, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.StatusResponse) => void): grpc.ClientUnaryCall;
    configure(request: nietzsche_db_pb.ConfigUpdate, metadata: grpc.Metadata, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.StatusResponse) => void): grpc.ClientUnaryCall;
    configure(request: nietzsche_db_pb.ConfigUpdate, metadata: grpc.Metadata, options: Partial<grpc.CallOptions>, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.StatusResponse) => void): grpc.ClientUnaryCall;
    replicate(request: nietzsche_db_pb.ReplicationRequest, options?: Partial<grpc.CallOptions>): grpc.ClientReadableStream<nietzsche_db_pb.ReplicationLog>;
    replicate(request: nietzsche_db_pb.ReplicationRequest, metadata?: grpc.Metadata, options?: Partial<grpc.CallOptions>): grpc.ClientReadableStream<nietzsche_db_pb.ReplicationLog>;
    getDigest(request: nietzsche_db_pb.DigestRequest, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.DigestResponse) => void): grpc.ClientUnaryCall;
    getDigest(request: nietzsche_db_pb.DigestRequest, metadata: grpc.Metadata, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.DigestResponse) => void): grpc.ClientUnaryCall;
    getDigest(request: nietzsche_db_pb.DigestRequest, metadata: grpc.Metadata, options: Partial<grpc.CallOptions>, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.DigestResponse) => void): grpc.ClientUnaryCall;
    rebuildIndex(request: nietzsche_db_pb.RebuildIndexRequest, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.StatusResponse) => void): grpc.ClientUnaryCall;
    rebuildIndex(request: nietzsche_db_pb.RebuildIndexRequest, metadata: grpc.Metadata, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.StatusResponse) => void): grpc.ClientUnaryCall;
    rebuildIndex(request: nietzsche_db_pb.RebuildIndexRequest, metadata: grpc.Metadata, options: Partial<grpc.CallOptions>, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.StatusResponse) => void): grpc.ClientUnaryCall;
}

export class DatabaseClient extends grpc.Client implements IDatabaseClient {
    constructor(address: string, credentials: grpc.ChannelCredentials, options?: Partial<grpc.ClientOptions>);
    public createCollection(request: nietzsche_db_pb.CreateCollectionRequest, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.StatusResponse) => void): grpc.ClientUnaryCall;
    public createCollection(request: nietzsche_db_pb.CreateCollectionRequest, metadata: grpc.Metadata, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.StatusResponse) => void): grpc.ClientUnaryCall;
    public createCollection(request: nietzsche_db_pb.CreateCollectionRequest, metadata: grpc.Metadata, options: Partial<grpc.CallOptions>, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.StatusResponse) => void): grpc.ClientUnaryCall;
    public deleteCollection(request: nietzsche_db_pb.DeleteCollectionRequest, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.StatusResponse) => void): grpc.ClientUnaryCall;
    public deleteCollection(request: nietzsche_db_pb.DeleteCollectionRequest, metadata: grpc.Metadata, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.StatusResponse) => void): grpc.ClientUnaryCall;
    public deleteCollection(request: nietzsche_db_pb.DeleteCollectionRequest, metadata: grpc.Metadata, options: Partial<grpc.CallOptions>, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.StatusResponse) => void): grpc.ClientUnaryCall;
    public listCollections(request: nietzsche_db_pb.Empty, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.ListCollectionsResponse) => void): grpc.ClientUnaryCall;
    public listCollections(request: nietzsche_db_pb.Empty, metadata: grpc.Metadata, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.ListCollectionsResponse) => void): grpc.ClientUnaryCall;
    public listCollections(request: nietzsche_db_pb.Empty, metadata: grpc.Metadata, options: Partial<grpc.CallOptions>, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.ListCollectionsResponse) => void): grpc.ClientUnaryCall;
    public getCollectionStats(request: nietzsche_db_pb.CollectionStatsRequest, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.CollectionStatsResponse) => void): grpc.ClientUnaryCall;
    public getCollectionStats(request: nietzsche_db_pb.CollectionStatsRequest, metadata: grpc.Metadata, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.CollectionStatsResponse) => void): grpc.ClientUnaryCall;
    public getCollectionStats(request: nietzsche_db_pb.CollectionStatsRequest, metadata: grpc.Metadata, options: Partial<grpc.CallOptions>, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.CollectionStatsResponse) => void): grpc.ClientUnaryCall;
    public insert(request: nietzsche_db_pb.InsertRequest, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.InsertResponse) => void): grpc.ClientUnaryCall;
    public insert(request: nietzsche_db_pb.InsertRequest, metadata: grpc.Metadata, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.InsertResponse) => void): grpc.ClientUnaryCall;
    public insert(request: nietzsche_db_pb.InsertRequest, metadata: grpc.Metadata, options: Partial<grpc.CallOptions>, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.InsertResponse) => void): grpc.ClientUnaryCall;
    public batchInsert(request: nietzsche_db_pb.BatchInsertRequest, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.InsertResponse) => void): grpc.ClientUnaryCall;
    public batchInsert(request: nietzsche_db_pb.BatchInsertRequest, metadata: grpc.Metadata, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.InsertResponse) => void): grpc.ClientUnaryCall;
    public batchInsert(request: nietzsche_db_pb.BatchInsertRequest, metadata: grpc.Metadata, options: Partial<grpc.CallOptions>, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.InsertResponse) => void): grpc.ClientUnaryCall;
    public insertText(request: nietzsche_db_pb.InsertTextRequest, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.InsertResponse) => void): grpc.ClientUnaryCall;
    public insertText(request: nietzsche_db_pb.InsertTextRequest, metadata: grpc.Metadata, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.InsertResponse) => void): grpc.ClientUnaryCall;
    public insertText(request: nietzsche_db_pb.InsertTextRequest, metadata: grpc.Metadata, options: Partial<grpc.CallOptions>, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.InsertResponse) => void): grpc.ClientUnaryCall;
    public delete(request: nietzsche_db_pb.DeleteRequest, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.DeleteResponse) => void): grpc.ClientUnaryCall;
    public delete(request: nietzsche_db_pb.DeleteRequest, metadata: grpc.Metadata, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.DeleteResponse) => void): grpc.ClientUnaryCall;
    public delete(request: nietzsche_db_pb.DeleteRequest, metadata: grpc.Metadata, options: Partial<grpc.CallOptions>, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.DeleteResponse) => void): grpc.ClientUnaryCall;
    public search(request: nietzsche_db_pb.SearchRequest, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.SearchResponse) => void): grpc.ClientUnaryCall;
    public search(request: nietzsche_db_pb.SearchRequest, metadata: grpc.Metadata, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.SearchResponse) => void): grpc.ClientUnaryCall;
    public search(request: nietzsche_db_pb.SearchRequest, metadata: grpc.Metadata, options: Partial<grpc.CallOptions>, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.SearchResponse) => void): grpc.ClientUnaryCall;
    public searchBatch(request: nietzsche_db_pb.BatchSearchRequest, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.BatchSearchResponse) => void): grpc.ClientUnaryCall;
    public searchBatch(request: nietzsche_db_pb.BatchSearchRequest, metadata: grpc.Metadata, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.BatchSearchResponse) => void): grpc.ClientUnaryCall;
    public searchBatch(request: nietzsche_db_pb.BatchSearchRequest, metadata: grpc.Metadata, options: Partial<grpc.CallOptions>, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.BatchSearchResponse) => void): grpc.ClientUnaryCall;
    public monitor(request: nietzsche_db_pb.MonitorRequest, options?: Partial<grpc.CallOptions>): grpc.ClientReadableStream<nietzsche_db_pb.SystemStats>;
    public monitor(request: nietzsche_db_pb.MonitorRequest, metadata?: grpc.Metadata, options?: Partial<grpc.CallOptions>): grpc.ClientReadableStream<nietzsche_db_pb.SystemStats>;
    public triggerSnapshot(request: nietzsche_db_pb.Empty, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.StatusResponse) => void): grpc.ClientUnaryCall;
    public triggerSnapshot(request: nietzsche_db_pb.Empty, metadata: grpc.Metadata, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.StatusResponse) => void): grpc.ClientUnaryCall;
    public triggerSnapshot(request: nietzsche_db_pb.Empty, metadata: grpc.Metadata, options: Partial<grpc.CallOptions>, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.StatusResponse) => void): grpc.ClientUnaryCall;
    public triggerVacuum(request: nietzsche_db_pb.Empty, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.StatusResponse) => void): grpc.ClientUnaryCall;
    public triggerVacuum(request: nietzsche_db_pb.Empty, metadata: grpc.Metadata, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.StatusResponse) => void): grpc.ClientUnaryCall;
    public triggerVacuum(request: nietzsche_db_pb.Empty, metadata: grpc.Metadata, options: Partial<grpc.CallOptions>, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.StatusResponse) => void): grpc.ClientUnaryCall;
    public configure(request: nietzsche_db_pb.ConfigUpdate, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.StatusResponse) => void): grpc.ClientUnaryCall;
    public configure(request: nietzsche_db_pb.ConfigUpdate, metadata: grpc.Metadata, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.StatusResponse) => void): grpc.ClientUnaryCall;
    public configure(request: nietzsche_db_pb.ConfigUpdate, metadata: grpc.Metadata, options: Partial<grpc.CallOptions>, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.StatusResponse) => void): grpc.ClientUnaryCall;
    public replicate(request: nietzsche_db_pb.ReplicationRequest, options?: Partial<grpc.CallOptions>): grpc.ClientReadableStream<nietzsche_db_pb.ReplicationLog>;
    public replicate(request: nietzsche_db_pb.ReplicationRequest, metadata?: grpc.Metadata, options?: Partial<grpc.CallOptions>): grpc.ClientReadableStream<nietzsche_db_pb.ReplicationLog>;
    public getDigest(request: nietzsche_db_pb.DigestRequest, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.DigestResponse) => void): grpc.ClientUnaryCall;
    public getDigest(request: nietzsche_db_pb.DigestRequest, metadata: grpc.Metadata, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.DigestResponse) => void): grpc.ClientUnaryCall;
    public getDigest(request: nietzsche_db_pb.DigestRequest, metadata: grpc.Metadata, options: Partial<grpc.CallOptions>, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.DigestResponse) => void): grpc.ClientUnaryCall;
    public rebuildIndex(request: nietzsche_db_pb.RebuildIndexRequest, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.StatusResponse) => void): grpc.ClientUnaryCall;
    public rebuildIndex(request: nietzsche_db_pb.RebuildIndexRequest, metadata: grpc.Metadata, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.StatusResponse) => void): grpc.ClientUnaryCall;
    public rebuildIndex(request: nietzsche_db_pb.RebuildIndexRequest, metadata: grpc.Metadata, options: Partial<grpc.CallOptions>, callback: (error: grpc.ServiceError | null, response: nietzsche_db_pb.StatusResponse) => void): grpc.ClientUnaryCall;
}
