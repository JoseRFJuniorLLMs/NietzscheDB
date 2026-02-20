.PHONY: proto test build vet

proto:
	protoc --go_out=. --go-grpc_out=. \
		--go_opt=module=nietzsche-sdk \
		--go-grpc_opt=module=nietzsche-sdk \
		proto/nietzsche.proto

test:
	go test -v -race ./...

build:
	go build ./...

vet:
	go vet ./...
