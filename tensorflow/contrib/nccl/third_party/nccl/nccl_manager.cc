#include "nccl_manager.h"
#include <mutex>

using std::lock_guard;
using std::mutex;

NCCLManager * NCCLManager::singleton = nullptr;
mutex NCCLManager::mu;

ncclComm_t NCCLManager::giveCommunicator(std::string deviceConfig)
{
    lock_guard<mutex> l(mu);
    if(communicator_storage.find(deviceConfig) == communicator_storage.end())
        return NULL;
    return communicator_storage[deviceConfig];
}

void NCCLManager::storeCommunicator(std::string deviceConfig, ncclComm_t comm)
{
    lock_guard<mutex> l(mu);
    communicator_storage[deviceConfig] = comm;
}

ncclUniqueId NCCLManager::giveNCCLId(std::string deviceConfig)
{
    lock_guard<mutex> l(mu);
    if(id_storage.find(deviceConfig) == id_storage.end())
    {
        ncclUniqueId id;
        ncclGetUniqueId(&id);
        id_storage[deviceConfig] = id;
    }
    return id_storage[deviceConfig];
}
