echo "\nHOW TO USE acl_test\n\n"
echo "./acl_test all [Perform all the experiment supported]"
echo "./acl_test backend[cl || neon] model[mobilenet || vgg16] dtype[float16 || float32]"
echo "\n"

echo "\nHOW TO USE acl_test via CK\n\n"
echo "ck run --cmd_key=all [Perform all the experiment supported]"
echo "ck run --cmd_key=run-net --env.CK_ACL_BACKEND=[cl || neon] --env.CK_ACL_MODEL=[mobilenet || vgg16] --env.CK_ACL_DTYPE=[float16 || float32]"
echo "\n"

