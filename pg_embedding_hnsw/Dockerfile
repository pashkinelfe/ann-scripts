FROM ann-benchmarks

# https://github.com/pgvector/pgvector/blob/master/Dockerfile

RUN git clone https://github.com/neondatabase/pg_embedding.git  /tmp/pgembedding

RUN DEBIAN_FRONTEND=noninteractive apt-get -y install tzdata
RUN apt-get update && apt-get install -y --no-install-recommends build-essential postgresql postgresql-server-dev-all postgresql-contrib
RUN sh -c 'echo "local all all trust" > /etc/postgresql/14/main/pg_hba.conf'
RUN cd /tmp/pgembedding && \
#	git config pull.ff only && \
#	git checkout hnsw && \
#	git pull && \
	make clean && \
	make OPTFLAGS="-march=native" && \
	make install && \
	rm -R /tmp/pgembedding

USER postgres
RUN service postgresql start && \
    psql -c "CREATE USER ann WITH ENCRYPTED PASSWORD 'ann'" && \
    psql -c "CREATE DATABASE ann" && \
    psql -c "GRANT ALL PRIVILEGES ON DATABASE ann TO ann" && \
    psql -d ann -c "CREATE EXTENSION embedding" && \
    psql -d ann -c "CREATE EXTENSION pg_prewarm" && \
    psql -c "ALTER USER ann SET maintenance_work_mem = '16GB'"
USER root

RUN pip install psycopg[binary] pgvector
