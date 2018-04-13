/// contains type defns of essential functions such as Conv2d

use typing::ItemType;

mod conv;

trait Op {
    fn get_name() -> String;
    fn get_type() -> ItemType;
}