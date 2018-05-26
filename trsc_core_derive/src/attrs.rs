use syn;
use std::collections::BTreeMap;

pub fn get_is_stateful(attrs: &[syn::Attribute]) -> bool {
    for attr in attrs.iter() {
        if let syn::MetaItem::Word(id) = &attr.value {
            if "stateful" == id.as_ref() {
                return true;
            }
        }
    }
    return false;
}

pub fn get_fns(attrs: &[syn::Attribute]) -> BTreeMap<&str, String> {
    let mut map = BTreeMap::new();
    for attr in attrs.iter() {
        if let syn::MetaItem::NameValue(key, val) = &attr.value {
            let key = key.as_ref();
            if "path" == key { continue; }
            let val = if let syn::Lit::Str(s,..) = val { s }
                      else { continue };

            map.insert(key, val.clone());
        }
    }
    map
}

pub fn get_op_name(attrs: &[syn::Attribute]) -> Option<&str> {
    get_str_attr("name", attrs)
}

pub fn get_path(attrs: &[syn::Attribute]) -> Option<&str> {
    get_str_attr("path", attrs)
}

pub fn get_str_attr<'a>(keyname: &'static str, attrs: &'a [syn::Attribute]) -> Option<&'a str> {
    for attr in attrs.iter() {
        if let syn::MetaItem::NameValue(key, val) = &attr.value {
            let key = key.as_ref();
            if keyname == key { 
                if let syn::Lit::Str(s,..) = val { return Some(s) }
                else { return None };
            }
        }
    }
    None
}
