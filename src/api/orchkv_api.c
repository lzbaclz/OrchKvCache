/* orchkv_api.c – public C API (stub, full impl in A8) */
#include "orchkv_api.h"

int orchkv_init(const orchkv_config_t *config)
{
    (void)config;
    LOG_INFO("orchkv_init called (stub)");
    return ORCHKV_OK;
}

int orchkv_shutdown(void)
{
    LOG_INFO("orchkv_shutdown called (stub)");
    return ORCHKV_OK;
}
