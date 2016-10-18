#ifndef NCCL_MANAGER_H
#define NCCL_MANAGER_H

#include <map>
#include <string>
#include <mutex>
#include "third_party/nccl/nccl.h"

using std::mutex;
using std::lock_guard;

class NCCLManager
{
    private:
        std::map<std::string, ncclComm_t> communicator_storage;
        std::map<std::string, ncclUniqueId> id_storage;
        static mutex mu;
        static NCCLManager * singleton;
        NCCLManager() {}
    public:
        ncclComm_t giveCommunicator(std::string deviceConfig);
        void storeCommunicator(std::string deviceConfig, ncclComm_t comm);
        ncclUniqueId giveNCCLId(std::string deviceConfig);

        static NCCLManager * getManager()
        {
            lock_guard<mutex> l(mu);
            if(!singleton)
            {
                singleton = new NCCLManager;
            }
            return singleton;
        }
};
#endif
