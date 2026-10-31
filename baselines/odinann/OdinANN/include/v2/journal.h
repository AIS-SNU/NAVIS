#ifndef DUMMY_KVS_H_
#define DUMMY_KVS_H_

#include <pthread.h>
// #include <rocksdb/utilities/optimistic_transaction_db.h>
// #include <rocksdb/utilities/transaction.h>
#include <string>
#include "utils.h"
#include "libcuckoo/cuckoohash_map.hh"

namespace v2 {
  // using namespace rocksdb;
  enum TxType { kInsert, kDelete };
  template<class TagT>
  class Journal {
   public:
    std::string db_name;
    // rocksdb::OptimisticTransactionDB *db;

    Journal(std::string path) : db_name(path) {
      // rocksdb::Options options;
      // options.create_if_missing = true;
      // rocksdb::Status status = rocksdb::OptimisticTransactionDB::Open(options, db_name, &db);
      // if (!status.ok()) {
      //   LOG(INFO) << "Failed to open db: " << db_name;
      //   exit(1);
      // }
    }
    ~Journal() {
      // delete db;
    }

    std::atomic<uint64_t> cur_txid;
    libcuckoo::cuckoohash_map<uint64_t, TagT> running_txs;

    std::string serialize(TxType type, TagT tag) {
      return std::to_string(type) + "_" + std::to_string(tag);
    }

    void deserialize(const std::string &s, TxType &type, TagT &tag) {
      // auto pos = s.find('_');
      // type = (TxType) std::stoi(s.substr(0, pos));
      // tag = (TagT) std::stoi(s.substr(pos + 1));
    }

    void append(TxType type, TagT tag) {
      // uint64_t txid = cur_txid.fetch_add(1);
      // running_txs.insert(txid, tag);
      // db->Put(rocksdb::WriteOptions(), std::to_string(txid), serialize(type, tag));
      // running_txs.erase(txid);
    }

    void checkpoint() {
      // auto locked_table = running_txs.lock_table();
      // uint64_t running_min_tx = std::numeric_limits<uint64_t>::max();
      // for (auto &item : locked_table) {
      //   running_min_tx = std::min(running_min_tx, item.first);
      // }
      // db->Put(rocksdb::WriteOptions(), "checkpoint", std::to_string(running_min_tx));
      // db->SyncWAL();
      // locked_table.unlock();
    }
  };
}  // namespace v2
#endif  // DUMMY_KVS_H_