NVCC = /opt/cuda/bin/nvcc
CFLAGS = -O2 -std=c++20
LDFLAGS = -lcudart -lpthread

p2pmark: p2pmark.cu
	$(NVCC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f p2pmark

.PHONY: clean
