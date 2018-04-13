use typing::ItemType;
use core::Op;

struct Conv2d;

impl Op for Conv2d {
    fn get_name() -> String {
        "Conv2d".to_owned()
    }

    fn get_type() -> ItemType {
        ItemType::Op
    }
}
